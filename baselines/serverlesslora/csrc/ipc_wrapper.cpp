#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <string>
#include <mutex>

namespace py = pybind11;

// --- Error Checking Macros ---
#define CUDA_DRV_CHECK(call) do {                         \
  CUresult _e = (call);                                   \
  if (_e != CUDA_SUCCESS) {                               \
    const char* _err = nullptr;                           \
    cuGetErrorString(_e, &_err);                          \
    throw std::runtime_error(std::string("CUDA DRV Error: ") + (_err ? _err : "unknown")); \
  }                                                       \
} while (0)

#define CUDA_RT_CHECK(call) do {                          \
  cudaError_t _e = (call);                                \
  if (_e != cudaSuccess) {                                \
    throw std::runtime_error(std::string("CUDA RT Error: ") + cudaGetErrorString(_e)); \
  }                                                       \
} while (0)

// Ensures cuInit is called exactly once.
void ensure_cuda_initialized() {
    static std::once_flag flag;
    std::call_once(flag, []{ CUDA_DRV_CHECK(cuInit(0)); });
}

// --- Functions Exposed to Python ---

// Returns (ipc_handle_bytes, dtype_code, offset_bytes, device_index, allocation_bytes)
py::tuple get_ipc_details(const at::Tensor& tensor) {
    if (!tensor.is_cuda()) {
        throw std::invalid_argument("Input tensor must be a CUDA tensor.");
    }
    ensure_cuda_initialized();
    const int device_index = tensor.get_device();

    CUdeviceptr ptr  = reinterpret_cast<CUdeviceptr>(tensor.data_ptr());
    CUdeviceptr base = 0;
    size_t alloc_bytes = 0;
    CUDA_DRV_CHECK(cuMemGetAddressRange(&base, &alloc_bytes, ptr));
    if (base == 0) {
        throw std::runtime_error("cuMemGetAddressRange failed to find the allocation base.");
    }

    CUipcMemHandle drv_handle;
    CUDA_DRV_CHECK(cuIpcGetMemHandle(&drv_handle, base));

    const int64_t offset_bytes = static_cast<int64_t>(ptr - base);
    const int dtype_code = static_cast<int>(tensor.scalar_type());
    py::bytes handle_bytes(reinterpret_cast<const char*>(&drv_handle), sizeof(drv_handle));

    return py::make_tuple(handle_bytes, dtype_code, offset_bytes, device_index, static_cast<int64_t>(alloc_bytes));
}

at::Tensor open_ipc_as_tensor(
    const py::bytes& handle_bytes,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    int dtype_code,
    int device_index,
    int64_t offset_bytes)
{
    if (PyBytes_Size(handle_bytes.ptr()) != static_cast<Py_ssize_t>(sizeof(CUipcMemHandle))) {
        throw std::invalid_argument("IPC handle has incorrect size.");
    }
    ensure_cuda_initialized();
    CUDA_RT_CHECK(cudaSetDevice(device_index));

    CUipcMemHandle drv_handle;
    std::memcpy(&drv_handle, PyBytes_AsString(handle_bytes.ptr()), sizeof(drv_handle));

    CUdeviceptr base_ptr = 0;
    CUDA_DRV_CHECK(cuIpcOpenMemHandle(&base_ptr, drv_handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));

    void* dev_ptr = reinterpret_cast<void*>(base_ptr + static_cast<size_t>(offset_bytes));
    auto scalar_type = static_cast<at::ScalarType>(dtype_code);
    auto options = at::TensorOptions().device(at::kCUDA, device_index).dtype(scalar_type);

    // Create a tensor view into the mapped memory. We pass a nullptr deleter
    // because this process does not own the memory.
    if (!strides.empty()) {
        return at::from_blob(dev_ptr, at::IntArrayRef(sizes), at::IntArrayRef(strides), nullptr, options);
    } else {
        return at::from_blob(dev_ptr, at::IntArrayRef(sizes), nullptr, options);
    }
}

// --- Pybind11 Module Definition ---
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "C++ extension for zero-copy tensor sharing via CUDA IPC.";
    m.def("get_ipc_details", &get_ipc_details, "Exports CUDA IPC details for a tensor.");
    m.def("open_ipc_as_tensor", &open_ipc_as_tensor, "Opens a CUDA IPC handle and reconstructs a tensor view.");
}
