/**
 * CUDA Stream Loader Extension for ServerlessLoRA
 *
 * Implements Section 5 of the paper:
 * - CUDA Streams for concurrent tensor loading
 * - CUDA Asynchronous Memory Transfer to overlap loading and GPU transferring
 *
 * This extension provides:
 * - Multi-stream tensor loading for parallel data transfer
 * - Async memory copy operations (H2D, D2D, D2H)
 * - Stream pool management for efficient resource reuse
 * - Batch loading with automatic stream assignment
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <mutex>
#include <memory>

namespace py = pybind11;

// --- Error Checking Macros ---
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        throw std::runtime_error(std::string("CUDA Error at ") +       \
            __FILE__ + ":" + std::to_string(__LINE__) + " - " +        \
            cudaGetErrorString(_e));                                   \
    }                                                                  \
} while (0)

#define CUDA_DRV_CHECK(call) do {                                      \
    CUresult _e = (call);                                              \
    if (_e != CUDA_SUCCESS) {                                          \
        const char* _err = nullptr;                                    \
        cuGetErrorString(_e, &_err);                                   \
        throw std::runtime_error(std::string("CUDA Driver Error: ") +  \
            (_err ? _err : "unknown"));                                \
    }                                                                  \
} while (0)

// Ensures cuInit is called exactly once
static void ensure_cuda_initialized() {
    static std::once_flag flag;
    std::call_once(flag, []{
        CUresult res = cuInit(0);
        if (res != CUDA_SUCCESS) {
            const char* err = nullptr;
            cuGetErrorString(res, &err);
            throw std::runtime_error(std::string("CUDA Init Error: ") + (err ? err : "unknown"));
        }
    });
}

// =============================================================================
// Stream Pool - Manages a pool of CUDA streams for efficient reuse
// =============================================================================

class StreamPool {
public:
    StreamPool(int device_id, int num_streams = 4)
        : device_id_(device_id), num_streams_(num_streams) {
        CUDA_CHECK(cudaSetDevice(device_id_));

        // Create streams with high priority for faster execution
        int least_priority, greatest_priority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));

        streams_.resize(num_streams_);
        for (int i = 0; i < num_streams_; ++i) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&streams_[i],
                cudaStreamNonBlocking, greatest_priority));
            available_streams_.push(i);
        }
    }

    ~StreamPool() {
        for (auto& stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }

    // Acquire a stream from the pool (blocking if none available)
    int acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (available_streams_.empty()) {
            // Wait for a stream to become available
            lock.unlock();
            std::this_thread::yield();
            lock.lock();
        }
        int idx = available_streams_.front();
        available_streams_.pop();
        return idx;
    }

    // Release a stream back to the pool
    void release(int idx) {
        std::lock_guard<std::mutex> lock(mutex_);
        available_streams_.push(idx);
    }

    cudaStream_t get(int idx) {
        return streams_[idx];
    }

    // Synchronize all streams
    void synchronize_all() {
        CUDA_CHECK(cudaSetDevice(device_id_));
        for (auto& stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }

    int device_id() const { return device_id_; }
    int num_streams() const { return num_streams_; }

private:
    int device_id_;
    int num_streams_;
    std::vector<cudaStream_t> streams_;
    std::queue<int> available_streams_;
    std::mutex mutex_;
};

// Global stream pools per device
static std::unordered_map<int, std::unique_ptr<StreamPool>> g_stream_pools;
static std::mutex g_pool_mutex;

StreamPool& get_stream_pool(int device_id, int num_streams = 4) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    auto it = g_stream_pools.find(device_id);
    if (it == g_stream_pools.end()) {
        g_stream_pools[device_id] = std::make_unique<StreamPool>(device_id, num_streams);
    }
    return *g_stream_pools[device_id];
}

// =============================================================================
// Async Memory Operations
// =============================================================================

/**
 * Asynchronously copy tensor from CPU to GPU using a stream
 * Returns immediately, use synchronize to wait for completion
 */
void async_copy_to_gpu(
    at::Tensor& dst,
    const at::Tensor& src,
    int stream_idx = -1)
{
    if (!src.is_cpu()) {
        throw std::invalid_argument("Source tensor must be on CPU");
    }
    if (!dst.is_cuda()) {
        throw std::invalid_argument("Destination tensor must be on CUDA");
    }
    if (src.numel() != dst.numel()) {
        throw std::invalid_argument("Source and destination must have same number of elements");
    }

    int device_id = dst.device().index();
    CUDA_CHECK(cudaSetDevice(device_id));

    StreamPool& pool = get_stream_pool(device_id);
    bool acquired = false;
    int idx = stream_idx;

    if (idx < 0) {
        idx = pool.acquire();
        acquired = true;
    }

    cudaStream_t stream = pool.get(idx);
    size_t size_bytes = src.numel() * src.element_size();

    // Use pinned memory for faster transfer if source is contiguous
    if (src.is_contiguous() && dst.is_contiguous()) {
        CUDA_CHECK(cudaMemcpyAsync(
            dst.data_ptr(),
            src.data_ptr(),
            size_bytes,
            cudaMemcpyHostToDevice,
            stream));
    } else {
        // Fallback: synchronous copy for non-contiguous tensors
        dst.copy_(src, true);  // non_blocking=true
    }

    if (acquired) {
        pool.release(idx);
    }
}

/**
 * Asynchronously copy tensor from GPU to CPU
 */
void async_copy_to_cpu(
    at::Tensor& dst,
    const at::Tensor& src,
    int stream_idx = -1)
{
    if (!src.is_cuda()) {
        throw std::invalid_argument("Source tensor must be on CUDA");
    }
    if (!dst.is_cpu()) {
        throw std::invalid_argument("Destination tensor must be on CPU");
    }

    int device_id = src.device().index();
    CUDA_CHECK(cudaSetDevice(device_id));

    StreamPool& pool = get_stream_pool(device_id);
    bool acquired = false;
    int idx = stream_idx;

    if (idx < 0) {
        idx = pool.acquire();
        acquired = true;
    }

    cudaStream_t stream = pool.get(idx);
    size_t size_bytes = src.numel() * src.element_size();

    if (src.is_contiguous() && dst.is_contiguous()) {
        CUDA_CHECK(cudaMemcpyAsync(
            dst.data_ptr(),
            src.data_ptr(),
            size_bytes,
            cudaMemcpyDeviceToHost,
            stream));
    } else {
        dst.copy_(src, true);
    }

    if (acquired) {
        pool.release(idx);
    }
}

/**
 * Asynchronously copy tensor between GPUs (or same GPU)
 */
void async_copy_gpu_to_gpu(
    at::Tensor& dst,
    const at::Tensor& src,
    int stream_idx = -1)
{
    if (!src.is_cuda() || !dst.is_cuda()) {
        throw std::invalid_argument("Both tensors must be on CUDA");
    }

    int src_device = src.device().index();
    int dst_device = dst.device().index();

    CUDA_CHECK(cudaSetDevice(dst_device));

    StreamPool& pool = get_stream_pool(dst_device);
    bool acquired = false;
    int idx = stream_idx;

    if (idx < 0) {
        idx = pool.acquire();
        acquired = true;
    }

    cudaStream_t stream = pool.get(idx);
    size_t size_bytes = src.numel() * src.element_size();

    if (src.is_contiguous() && dst.is_contiguous()) {
        if (src_device == dst_device) {
            CUDA_CHECK(cudaMemcpyAsync(
                dst.data_ptr(),
                src.data_ptr(),
                size_bytes,
                cudaMemcpyDeviceToDevice,
                stream));
        } else {
            // Peer-to-peer copy
            CUDA_CHECK(cudaMemcpyPeerAsync(
                dst.data_ptr(), dst_device,
                src.data_ptr(), src_device,
                size_bytes,
                stream));
        }
    } else {
        dst.copy_(src, true);
    }

    if (acquired) {
        pool.release(idx);
    }
}

// =============================================================================
// Batch Loading with Multiple Streams
// =============================================================================

/**
 * Load multiple tensors concurrently using multiple CUDA streams
 *
 * Paper Section 5: "utilize CUDA Streams to load tensors concurrently"
 *
 * This function distributes tensor loading across multiple streams
 * to maximize PCIe bandwidth utilization.
 */
std::vector<at::Tensor> batch_load_tensors_async(
    const std::vector<at::Tensor>& cpu_tensors,
    int device_id,
    int num_streams = 4)
{
    if (cpu_tensors.empty()) {
        return {};
    }

    ensure_cuda_initialized();
    CUDA_CHECK(cudaSetDevice(device_id));

    // Get or create stream pool
    StreamPool& pool = get_stream_pool(device_id, num_streams);

    std::vector<at::Tensor> gpu_tensors;
    gpu_tensors.reserve(cpu_tensors.size());

    // Allocate GPU tensors
    auto options = at::TensorOptions().device(at::kCUDA, device_id);
    for (const auto& cpu_tensor : cpu_tensors) {
        gpu_tensors.push_back(
            torch::empty(cpu_tensor.sizes(), options.dtype(cpu_tensor.dtype()))
        );
    }

    // Distribute copies across streams
    std::vector<int> stream_assignments(cpu_tensors.size());
    for (size_t i = 0; i < cpu_tensors.size(); ++i) {
        stream_assignments[i] = i % num_streams;
    }

    // Launch async copies
    for (size_t i = 0; i < cpu_tensors.size(); ++i) {
        const auto& src = cpu_tensors[i];
        auto& dst = gpu_tensors[i];
        int stream_idx = stream_assignments[i];

        cudaStream_t stream = pool.get(stream_idx);
        size_t size_bytes = src.numel() * src.element_size();

        if (src.is_contiguous()) {
            CUDA_CHECK(cudaMemcpyAsync(
                dst.data_ptr(),
                src.data_ptr(),
                size_bytes,
                cudaMemcpyHostToDevice,
                stream));
        } else {
            // For non-contiguous, make contiguous first
            auto src_contig = src.contiguous();
            CUDA_CHECK(cudaMemcpyAsync(
                dst.data_ptr(),
                src_contig.data_ptr(),
                size_bytes,
                cudaMemcpyHostToDevice,
                stream));
        }
    }

    // Synchronize all streams
    pool.synchronize_all();

    return gpu_tensors;
}

/**
 * Offload multiple tensors from GPU to CPU concurrently
 */
std::vector<at::Tensor> batch_offload_tensors_async(
    const std::vector<at::Tensor>& gpu_tensors,
    int num_streams = 4)
{
    if (gpu_tensors.empty()) {
        return {};
    }

    int device_id = gpu_tensors[0].device().index();
    CUDA_CHECK(cudaSetDevice(device_id));

    StreamPool& pool = get_stream_pool(device_id, num_streams);

    std::vector<at::Tensor> cpu_tensors;
    cpu_tensors.reserve(gpu_tensors.size());

    // Allocate pinned CPU tensors for faster transfer
    for (const auto& gpu_tensor : gpu_tensors) {
        // Use pinned memory for faster D2H transfer
        auto cpu_tensor = torch::empty(
            gpu_tensor.sizes(),
            gpu_tensor.options().device(at::kCPU).pinned_memory(true)
        );
        cpu_tensors.push_back(cpu_tensor);
    }

    // Launch async copies
    for (size_t i = 0; i < gpu_tensors.size(); ++i) {
        const auto& src = gpu_tensors[i];
        auto& dst = cpu_tensors[i];
        int stream_idx = i % num_streams;

        cudaStream_t stream = pool.get(stream_idx);
        size_t size_bytes = src.numel() * src.element_size();

        if (src.is_contiguous()) {
            CUDA_CHECK(cudaMemcpyAsync(
                dst.data_ptr(),
                src.data_ptr(),
                size_bytes,
                cudaMemcpyDeviceToHost,
                stream));
        } else {
            auto src_contig = src.contiguous();
            CUDA_CHECK(cudaMemcpyAsync(
                dst.data_ptr(),
                src_contig.data_ptr(),
                size_bytes,
                cudaMemcpyDeviceToHost,
                stream));
        }
    }

    pool.synchronize_all();

    return cpu_tensors;
}

// =============================================================================
// Stream Management Functions
// =============================================================================

/**
 * Initialize stream pool for a device
 */
void init_stream_pool(int device_id, int num_streams = 4) {
    ensure_cuda_initialized();
    get_stream_pool(device_id, num_streams);
}

/**
 * Synchronize all streams on a device
 */
void synchronize_streams(int device_id) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    auto it = g_stream_pools.find(device_id);
    if (it != g_stream_pools.end()) {
        it->second->synchronize_all();
    }
}

/**
 * Synchronize all streams on all devices
 */
void synchronize_all_streams() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    for (auto& pair : g_stream_pools) {
        pair.second->synchronize_all();
    }
}

/**
 * Get stream pool info
 */
py::dict get_stream_pool_info(int device_id) {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    py::dict info;

    auto it = g_stream_pools.find(device_id);
    if (it != g_stream_pools.end()) {
        info["device_id"] = it->second->device_id();
        info["num_streams"] = it->second->num_streams();
        info["initialized"] = true;
    } else {
        info["device_id"] = device_id;
        info["num_streams"] = 0;
        info["initialized"] = false;
    }

    return info;
}

/**
 * Cleanup stream pools (call before exit)
 */
void cleanup_stream_pools() {
    std::lock_guard<std::mutex> lock(g_pool_mutex);
    g_stream_pools.clear();
}

// =============================================================================
// Memory Prefetch (for unified memory)
// =============================================================================

/**
 * Prefetch tensor data to GPU asynchronously
 * Works with managed memory (CUDA Unified Memory)
 */
void prefetch_to_gpu(at::Tensor& tensor, int device_id, int stream_idx = -1) {
    if (!tensor.is_cuda()) {
        throw std::invalid_argument("Tensor must be on CUDA");
    }

    CUDA_CHECK(cudaSetDevice(device_id));

    StreamPool& pool = get_stream_pool(device_id);
    int idx = (stream_idx >= 0) ? stream_idx : 0;
    cudaStream_t stream = pool.get(idx);

    // Note: cudaMemPrefetchAsync only works with managed memory
    // For regular device memory, this is a no-op
    // CUDA 13 replaced the plain `int dstDevice` parameter with a
    // cudaMemLocation struct; keep both call forms so this builds against
    // either toolkit version.
#if CUDART_VERSION >= 13000
    cudaMemLocation loc;
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device_id;
    cudaError_t err = cudaMemPrefetchAsync(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        loc,
        0,
        stream
    );
#else
    cudaError_t err = cudaMemPrefetchAsync(
        tensor.data_ptr(),
        tensor.numel() * tensor.element_size(),
        device_id,
        stream
    );
#endif

    // Ignore error if not managed memory
    if (err != cudaSuccess && err != cudaErrorInvalidValue) {
        CUDA_CHECK(err);
    }
}

// =============================================================================
// Pybind11 Module Definition
// =============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = R"pbdoc(
        CUDA Stream Loader Extension for ServerlessLoRA

        Provides concurrent tensor loading using CUDA Streams and
        asynchronous memory transfer for accelerated pre-loading.

        Paper Section 5: "utilize CUDA Streams to load tensors concurrently
        and CUDA Asynchronous Memory Transfer to overlap loading and GPU transferring"
    )pbdoc";

    // Stream pool management
    m.def("init_stream_pool", &init_stream_pool,
          py::arg("device_id"), py::arg("num_streams") = 4,
          "Initialize CUDA stream pool for a device");

    m.def("synchronize_streams", &synchronize_streams,
          py::arg("device_id"),
          "Synchronize all streams on a device");

    m.def("synchronize_all_streams", &synchronize_all_streams,
          "Synchronize all streams on all devices");

    m.def("get_stream_pool_info", &get_stream_pool_info,
          py::arg("device_id"),
          "Get stream pool information for a device");

    m.def("cleanup_stream_pools", &cleanup_stream_pools,
          "Cleanup all stream pools");

    // Single tensor async operations
    m.def("async_copy_to_gpu", &async_copy_to_gpu,
          py::arg("dst"), py::arg("src"), py::arg("stream_idx") = -1,
          "Asynchronously copy tensor from CPU to GPU");

    m.def("async_copy_to_cpu", &async_copy_to_cpu,
          py::arg("dst"), py::arg("src"), py::arg("stream_idx") = -1,
          "Asynchronously copy tensor from GPU to CPU");

    m.def("async_copy_gpu_to_gpu", &async_copy_gpu_to_gpu,
          py::arg("dst"), py::arg("src"), py::arg("stream_idx") = -1,
          "Asynchronously copy tensor between GPUs");

    // Batch operations
    m.def("batch_load_tensors_async", &batch_load_tensors_async,
          py::arg("cpu_tensors"), py::arg("device_id"), py::arg("num_streams") = 4,
          "Load multiple tensors to GPU concurrently using multiple streams");

    m.def("batch_offload_tensors_async", &batch_offload_tensors_async,
          py::arg("gpu_tensors"), py::arg("num_streams") = 4,
          "Offload multiple tensors from GPU to CPU concurrently");

    // Memory prefetch
    m.def("prefetch_to_gpu", &prefetch_to_gpu,
          py::arg("tensor"), py::arg("device_id"), py::arg("stream_idx") = -1,
          "Prefetch tensor data to GPU (for unified memory)");
}
