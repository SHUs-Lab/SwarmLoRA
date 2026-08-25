// Worker-side IPC: slot management, P2P copies to/from aggregator buffers.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <stdexcept>
#include <cstring>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

// Shared types and constants
#include "common/ipc_types.h"

namespace py = pybind11;

// Worker-specific constants
constexpr int MAX_SLOTS_PER_WORKER = 8;

// ===== SLOT-BASED DATA STRUCTURES =====

// Per-slot context (worker-side)
struct SlotContext {
    int slot_id;
    bool is_open;
    void* input_ptr;      // IPC-mapped input buffer
    void* output_ptr;     // IPC-mapped output buffer
    size_t input_buffer_size;
    size_t output_buffer_size;
    int64_t current_bx, current_tx, current_dx;
};

// Worker slot manager
struct WorkerSlotManager {
    int worker_id;
    int worker_device;
    int aggregator_device;

    SlotContext owned_slots[MAX_SLOTS_PER_WORKER];
    int num_owned_slots;
    bool initialized;

    // Model dimensions (set during initialization)
    int hidden_dim;
    int intermediate_dim;
    int qkv_dim;

    // Per-slot IPC pointers tracked in SlotContext (no shared base pointer)
};

static WorkerSlotManager g_worker_slots;


// ===== SLOT-BASED WORKER FUNCTIONS =====

void init_worker_slot_manager(
    int worker_id,
    int worker_device,
    int aggregator_device,
    int hidden_dim,
    int intermediate_dim,
    int qkv_dim
) {
    g_worker_slots.worker_id = worker_id;
    g_worker_slots.worker_device = worker_device;
    g_worker_slots.aggregator_device = aggregator_device;
    g_worker_slots.hidden_dim = hidden_dim;
    g_worker_slots.intermediate_dim = intermediate_dim;
    g_worker_slots.qkv_dim = qkv_dim;
    g_worker_slots.num_owned_slots = 0;

    // Initialize all slot contexts
    for (int i = 0; i < MAX_SLOTS_PER_WORKER; ++i) {
        g_worker_slots.owned_slots[i].slot_id = -1;
        g_worker_slots.owned_slots[i].is_open = false;
        g_worker_slots.owned_slots[i].input_ptr = nullptr;
        g_worker_slots.owned_slots[i].output_ptr = nullptr;
    }

    // Set device context
    CUDA_RT_CHECK(cudaSetDevice(worker_device));

    // P2P access is enabled lazily via cudaIpcMemLazyEnablePeerAccess in open_slot()
    // Explicit cudaDeviceEnablePeerAccess removed — it serializes in the CUDA driver
    // and costs ~0.1s/worker for cross-device workers during burst init

    g_worker_slots.initialized = true;
    printf("[Worker-%d] Slot manager initialized (worker_device=%d, aggregator_device=%d)\n",
           worker_id, worker_device, aggregator_device);
}


void open_slot(int slot_id, py::bytes input_handle, py::bytes output_handle,
               size_t input_buffer_size, size_t output_buffer_size) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("open_slot: worker slot manager not initialized");
    }
    if (g_worker_slots.num_owned_slots >= MAX_SLOTS_PER_WORKER) {
        throw std::runtime_error("open_slot: max slots per worker exceeded");
    }

    CUDA_RT_CHECK(cudaSetDevice(g_worker_slots.worker_device));

    // Find free slot context
    int ctx_idx = -1;
    for (int i = 0; i < MAX_SLOTS_PER_WORKER; ++i) {
        if (!g_worker_slots.owned_slots[i].is_open) {
            ctx_idx = i;
            break;
        }
    }
    if (ctx_idx < 0) {
        throw std::runtime_error("open_slot: no free slot context");
    }

    auto& ctx = g_worker_slots.owned_slots[ctx_idx];
    ctx.slot_id = slot_id;
    ctx.input_buffer_size = input_buffer_size;
    ctx.output_buffer_size = output_buffer_size;

    // Validate IPC handle sizes before memcpy to prevent buffer over-read
    cudaIpcMemHandle_t input_h, output_h;
    size_t in_sz  = static_cast<size_t>(PyBytes_Size(input_handle.ptr()));
    size_t out_sz = static_cast<size_t>(PyBytes_Size(output_handle.ptr()));
    if (in_sz != sizeof(cudaIpcMemHandle_t) || out_sz != sizeof(cudaIpcMemHandle_t)) {
        throw std::runtime_error(
            "open_slot: invalid IPC handle size (got " + std::to_string(in_sz) +
            "/" + std::to_string(out_sz) + ", expected " +
            std::to_string(sizeof(cudaIpcMemHandle_t)) + ")");
    }
    std::memcpy(&input_h,  PyBytes_AsString(input_handle.ptr()),  sizeof(cudaIpcMemHandle_t));
    std::memcpy(&output_h, PyBytes_AsString(output_handle.ptr()), sizeof(cudaIpcMemHandle_t));

    // Open handles; on failure clean up to avoid closing a null pointer
    cudaError_t in_err = cudaIpcOpenMemHandle(&ctx.input_ptr, input_h, cudaIpcMemLazyEnablePeerAccess);
    if (in_err != cudaSuccess) {
        ctx.input_ptr = nullptr;
        throw std::runtime_error(std::string("open_slot: input IPC open failed: ") + cudaGetErrorString(in_err));
    }
    cudaError_t out_err = cudaIpcOpenMemHandle(&ctx.output_ptr, output_h, cudaIpcMemLazyEnablePeerAccess);
    if (out_err != cudaSuccess) {
        cudaIpcCloseMemHandle(ctx.input_ptr);
        ctx.input_ptr = nullptr;
        ctx.output_ptr = nullptr;
        throw std::runtime_error(std::string("open_slot: output IPC open failed: ") + cudaGetErrorString(out_err));
    }

    ctx.current_bx = 0;
    ctx.current_tx = 0;
    ctx.current_dx = 0;
    ctx.is_open = true;
    g_worker_slots.num_owned_slots++;

    printf("[Worker-%d] Opened slot %d (input=%zu bytes, output=%zu bytes)\n",
           g_worker_slots.worker_id, slot_id, input_buffer_size, output_buffer_size);
}


void close_slot(int slot_id) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("close_slot: worker slot manager not initialized");
    }

    // Find slot context
    SlotContext* ctx = nullptr;
    for (int i = 0; i < MAX_SLOTS_PER_WORKER; ++i) {
        if (g_worker_slots.owned_slots[i].is_open &&
            g_worker_slots.owned_slots[i].slot_id == slot_id) {
            ctx = &g_worker_slots.owned_slots[i];
            break;
        }
    }
    if (!ctx) {
        throw std::runtime_error("close_slot: slot " + std::to_string(slot_id) + " not found");
    }

    // CRITICAL: Full device sync before releasing slot
    // CRITICAL: Full device sync before closing IPC handles
    // Device sync (not stream sync) is required here because:
    // 1. IPC handles may be used by multiple streams
    // 2. The aggregator may have pending writes on its stream
    // 3. This prevents race conditions when the slot is quickly re-claimed
    CUDA_RT_CHECK(cudaDeviceSynchronize());

    // Close per-slot IPC handles
    if (ctx->input_ptr) {
        cudaError_t err = cudaIpcCloseMemHandle(ctx->input_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Worker-%d] Warning: cudaIpcCloseMemHandle(input) for slot %d: %s\n",
                    g_worker_slots.worker_id, ctx->slot_id, cudaGetErrorString(err));
        }
        ctx->input_ptr = nullptr;
    }
    if (ctx->output_ptr) {
        cudaError_t err = cudaIpcCloseMemHandle(ctx->output_ptr);
        if (err != cudaSuccess) {
            fprintf(stderr, "[Worker-%d] Warning: cudaIpcCloseMemHandle(output) for slot %d: %s\n",
                    g_worker_slots.worker_id, ctx->slot_id, cudaGetErrorString(err));
        }
        ctx->output_ptr = nullptr;
    }
    ctx->slot_id = -1;
    ctx->is_open = false;
    g_worker_slots.num_owned_slots--;

    printf("[Worker-%d] Closed slot %d\n", g_worker_slots.worker_id, slot_id);
}


static SlotContext* get_slot_context(int slot_id) {
    for (int i = 0; i < MAX_SLOTS_PER_WORKER; ++i) {
        if (g_worker_slots.owned_slots[i].is_open &&
            g_worker_slots.owned_slots[i].slot_id == slot_id) {
            return &g_worker_slots.owned_slots[i];
        }
    }
    return nullptr;
}


void prepare_input_slot(int slot_id, int op_type, torch::Tensor input) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("prepare_input_slot: worker slot manager not initialized");
    }

    SlotContext* ctx = get_slot_context(slot_id);
    if (!ctx) {
        throw std::runtime_error("prepare_input_slot: slot " + std::to_string(slot_id) + " not found");
    }

    if (input.dim() != 3) {
        throw std::runtime_error("Input must be 3D [B, T, D]");
    }

    int64_t bx = input.size(0);
    int64_t tx = input.size(1);
    int64_t dx = input.size(2);

    // Calculate actual input size for this tensor
    size_t input_bytes = input.numel() * input.element_size();

    // Validate buffer size to prevent overflow
    if (input_bytes > ctx->input_buffer_size) {
        throw std::runtime_error(
            "prepare_input_slot: input tensor (" + std::to_string(input_bytes) +
            " bytes) exceeds slot buffer size (" + std::to_string(ctx->input_buffer_size) + " bytes)"
        );
    }

    // Copy input to slot's input buffer via P2P
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g_worker_slots.worker_device).stream();

    CUDA_RT_CHECK(cudaMemcpyAsync(
        ctx->input_ptr,
        input.data_ptr(),
        input_bytes,
        cudaMemcpyDefault,
        stream
    ));

    // Synchronize to ensure copy completes
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    // Store shape for subsequent get_outputs_slot
    ctx->current_bx = bx;
    ctx->current_tx = tx;
    ctx->current_dx = dx;

}


std::vector<torch::Tensor> get_outputs_slot(int slot_id, std::vector<int64_t> out_dims,
                                             torch::Tensor input_template) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("get_outputs_slot: worker slot manager not initialized");
    }

    SlotContext* ctx = get_slot_context(slot_id);
    if (!ctx) {
        throw std::runtime_error("get_outputs_slot: slot " + std::to_string(slot_id) + " not found");
    }

    auto options = input_template.options();
    int64_t bx = ctx->current_bx;
    int64_t tx = ctx->current_tx;
    size_t elem_size = input_template.element_size();

    // Calculate total output dimension
    int64_t total_out_dim = 0;
    for (auto d : out_dims) total_out_dim += d;
    size_t output_bytes = bx * tx * total_out_dim * elem_size;

    // Validate output buffer size to prevent overflow
    if (output_bytes > ctx->output_buffer_size) {
        throw std::runtime_error(
            "get_outputs_slot: output (" + std::to_string(output_bytes) +
            " bytes) exceeds slot output buffer (" + std::to_string(ctx->output_buffer_size) + " bytes)"
        );
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g_worker_slots.worker_device).stream();

    std::vector<torch::Tensor> outputs;

    if (out_dims.size() > 1) {
        // FUSED OPS: Copy entire concatenated output, then split
        torch::Tensor concat_out = torch::empty({bx, tx, total_out_dim}, options);

        CUDA_RT_CHECK(cudaMemcpyAsync(
        concat_out.data_ptr(),
        ctx->output_ptr,
        output_bytes,
        cudaMemcpyDefault,
        stream
    ));

        // No sync needed: P2P copy and subsequent PyTorch ops are on the same
        // worker stream, so GPU ordering is guaranteed. split_with_sizes is a
        // view operation (no data access), and downstream ops (LoRA add, etc.)
        // will naturally wait for the copy to complete on the same stream.

        // Split on the last dimension
        outputs = concat_out.split_with_sizes(
            c10::IntArrayRef(out_dims.data(), out_dims.size()), -1);
    } else {
        // Single output
        torch::Tensor out = torch::empty({bx, tx, out_dims[0]}, options);

        CUDA_RT_CHECK(cudaMemcpyAsync(
        out.data_ptr(),
        ctx->output_ptr,
        output_bytes,
        cudaMemcpyDefault,
        stream
    ));

        // No sync needed: same-stream ordering guarantees correctness.

        outputs.push_back(out);
    }

    return outputs;
}


void prepare_token_ids_slot(int slot_id, torch::Tensor token_ids) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("prepare_token_ids_slot: worker slot manager not initialized");
    }

    SlotContext* ctx = get_slot_context(slot_id);
    if (!ctx) {
        throw std::runtime_error("prepare_token_ids_slot: slot " + std::to_string(slot_id) + " not found");
    }

    if (token_ids.dim() != 2) {
        throw std::runtime_error("token_ids must be 2D [B, T]");
    }

    int64_t bx = token_ids.size(0);
    int64_t tx = token_ids.size(1);

    size_t bytes = token_ids.numel() * sizeof(int64_t);

    // Validate buffer size to prevent overflow
    if (bytes > ctx->input_buffer_size) {
        throw std::runtime_error(
            "prepare_token_ids_slot: token_ids (" + std::to_string(bytes) +
            " bytes) exceeds slot buffer size (" + std::to_string(ctx->input_buffer_size) + " bytes)"
        );
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g_worker_slots.worker_device).stream();

    CUDA_RT_CHECK(cudaMemcpyAsync(
        ctx->input_ptr,
        token_ids.data_ptr(),
        bytes,
        cudaMemcpyDefault,
        stream
    ));

    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    ctx->current_bx = bx;
    ctx->current_tx = tx;
    ctx->current_dx = g_worker_slots.hidden_dim;

}


torch::Tensor get_embeddings_slot(int slot_id, at::ScalarType dtype) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("get_embeddings_slot: worker slot manager not initialized");
    }

    SlotContext* ctx = get_slot_context(slot_id);
    if (!ctx) {
        throw std::runtime_error("get_embeddings_slot: slot " + std::to_string(slot_id) + " not found");
    }

    int64_t bx = ctx->current_bx;
    int64_t tx = ctx->current_tx;
    int hidden_dim = g_worker_slots.hidden_dim;

    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::Device(torch::kCUDA, g_worker_slots.worker_device));

    torch::Tensor embeddings = torch::empty({bx, tx, hidden_dim}, options);

    size_t elem_size = embeddings.element_size();
    size_t bytes = bx * tx * hidden_dim * elem_size;

    // Validate output buffer size to prevent overflow
    if (bytes > ctx->output_buffer_size) {
        throw std::runtime_error(
            "get_embeddings_slot: output (" + std::to_string(bytes) +
            " bytes) exceeds slot output buffer (" + std::to_string(ctx->output_buffer_size) + " bytes)"
        );
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g_worker_slots.worker_device).stream();

    CUDA_RT_CHECK(cudaMemcpyAsync(
        embeddings.data_ptr(),
        ctx->output_ptr,
        bytes,
        cudaMemcpyDefault,
        stream
    ));

    // No sync needed: same-stream ordering guarantees correctness.

    return embeddings;
}


void prepare_hidden_for_lm_head_slot(int slot_id, torch::Tensor hidden_states) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("prepare_hidden_for_lm_head_slot: worker slot manager not initialized");
    }

    SlotContext* ctx = get_slot_context(slot_id);
    if (!ctx) {
        throw std::runtime_error("prepare_hidden_for_lm_head_slot: slot " + std::to_string(slot_id) + " not found");
    }

    if (hidden_states.dim() != 3) {
        throw std::runtime_error("hidden_states must be 3D [B, T, D]");
    }

    int64_t bx = hidden_states.size(0);
    int64_t tx = hidden_states.size(1);
    int64_t dx = hidden_states.size(2);

    size_t bytes = hidden_states.numel() * hidden_states.element_size();

    // Validate buffer size to prevent overflow
    if (bytes > ctx->input_buffer_size) {
        throw std::runtime_error(
            "prepare_hidden_for_lm_head_slot: hidden_states (" + std::to_string(bytes) +
            " bytes) exceeds slot buffer size (" + std::to_string(ctx->input_buffer_size) + " bytes)"
        );
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g_worker_slots.worker_device).stream();

    CUDA_RT_CHECK(cudaMemcpyAsync(
        ctx->input_ptr,
        hidden_states.data_ptr(),
        bytes,
        cudaMemcpyDefault,
        stream
    ));

    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    ctx->current_bx = bx;
    ctx->current_tx = tx;
    ctx->current_dx = dx;
}


int64_t get_next_token_slot(int slot_id) {
    if (!g_worker_slots.initialized) {
        throw std::runtime_error("get_next_token_slot: worker slot manager not initialized");
    }

    SlotContext* ctx = get_slot_context(slot_id);
    if (!ctx) {
        throw std::runtime_error("get_next_token_slot: slot " + std::to_string(slot_id) + " not found");
    }

    // Stream sync is sufficient here: aggregator does cudaStreamSynchronize + epoch++
    // (memory_order_release) before signaling done. Worker's wait_for_epoch uses
    // memory_order_acquire, establishing happens-before. Only worker's own stream needs draining.
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(g_worker_slots.worker_device).stream();
    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    int64_t next_token;

    // Copy from output buffer (first int64)
    CUDA_RT_CHECK(cudaMemcpyAsync(
        &next_token, ctx->output_ptr,
        sizeof(int64_t),
        cudaMemcpyDefault,
        stream
    ));

    CUDA_RT_CHECK(cudaStreamSynchronize(stream));

    return next_token;
}


void cleanup_worker_slots() {
    if (!g_worker_slots.initialized) {
        return;
    }

    // Ensure all GPU operations are complete before closing IPC handles
    cudaSetDevice(g_worker_slots.worker_device);
    cudaDeviceSynchronize();

    // Close per-slot IPC handles
    for (int i = 0; i < MAX_SLOTS_PER_WORKER; ++i) {
        auto& ctx = g_worker_slots.owned_slots[i];
        if (ctx.is_open) {
            if (ctx.input_ptr) {
                cudaError_t err = cudaIpcCloseMemHandle(ctx.input_ptr);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[Worker-%d] Warning: cleanup cudaIpcCloseMemHandle(input slot %d): %s\n",
                            g_worker_slots.worker_id, ctx.slot_id, cudaGetErrorString(err));
                }
                ctx.input_ptr = nullptr;
            }
            if (ctx.output_ptr) {
                cudaError_t err = cudaIpcCloseMemHandle(ctx.output_ptr);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[Worker-%d] Warning: cleanup cudaIpcCloseMemHandle(output slot %d): %s\n",
                            g_worker_slots.worker_id, ctx.slot_id, cudaGetErrorString(err));
                }
                ctx.output_ptr = nullptr;
            }
            ctx.slot_id = -1;
            ctx.is_open = false;
        }
    }

    g_worker_slots.num_owned_slots = 0;
    g_worker_slots.initialized = false;
}


// Python module bindings

PYBIND11_MODULE(ext_unified_barrier, m) {
    m.doc() = "Slot-Based Worker Operations for Multi-GPU Inference";

    // Export operation type constants
    m.attr("OP_QKV_FUSED") = OP_QKV_FUSED;
    m.attr("OP_O_PROJ") = OP_O_PROJ;
    m.attr("OP_GATE_UP_FUSED") = OP_GATE_UP_FUSED;
    m.attr("OP_DOWN_PROJ") = OP_DOWN_PROJ;
    m.attr("OP_EMBED") = OP_EMBED;
    m.attr("OP_LM_HEAD") = OP_LM_HEAD;
    m.attr("OP_SET_SAMPLING") = OP_SET_SAMPLING;

    // ===== SLOT-BASED WORKER FUNCTIONS =====
    m.def("init_worker_slot_manager", &init_worker_slot_manager,
          py::arg("worker_id"),
          py::arg("worker_device"),
          py::arg("aggregator_device"),
          py::arg("hidden_dim"),
          py::arg("intermediate_dim"),
          py::arg("qkv_dim"),
          "Initialize worker slot manager for slot-based operations");

    m.def("open_slot", &open_slot,
          py::arg("slot_id"),
          py::arg("input_handle"),
          py::arg("output_handle"),
          py::arg("input_buffer_size"),
          py::arg("output_buffer_size"),
          "Open a slot with IPC handles");

    m.def("close_slot", &close_slot,
          py::arg("slot_id"),
          "Close a previously opened slot");

    m.def("prepare_input_slot", &prepare_input_slot,
          py::arg("slot_id"),
          py::arg("op_type"),
          py::arg("input"),
          "Write input to slot's input buffer (returns shape dict)");

    m.def("get_outputs_slot", &get_outputs_slot,
          py::arg("slot_id"),
          py::arg("out_dims"),
          py::arg("input_template"),
          "Read multiple outputs (for fused ops) from slot's output buffer");

    m.def("prepare_token_ids_slot", &prepare_token_ids_slot,
          py::arg("slot_id"),
          py::arg("token_ids"),
          "Write token IDs to slot for embedding lookup");

    m.def("get_embeddings_slot", &get_embeddings_slot,
          py::arg("slot_id"),
          py::arg("dtype"),
          "Read embeddings from slot after embed op");

    m.def("prepare_hidden_for_lm_head_slot", &prepare_hidden_for_lm_head_slot,
          py::arg("slot_id"),
          py::arg("hidden_states"),
          "Write hidden states to slot for LM head");

    m.def("get_next_token_slot", &get_next_token_slot,
          py::arg("slot_id"),
          "Read next token from slot after LM head op");

    m.def("cleanup_worker_slots", &cleanup_worker_slots,
          "Cleanup all worker slot resources");

}
