// Batched GEMM aggregator: slot pool, weight fusion, barrier-driven main loop.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <map>
#include <tuple>
#include <stdexcept>
#include <atomic>
#include <limits>
#include <chrono>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// Shared types and constants
#include "common/ipc_types.h"

// Verify that the worker submitting a command actually owns the target slot.
// Called after slot_id bounds check — prevents cross-slot activation theft.
#define VERIFY_SLOT_OWNER(slot_id, cmd_worker_id, ctx)                              \
    do {                                                                             \
        int _owner = g_slot_pool.slots[(slot_id)].owner_worker_id                  \
                         .load(std::memory_order_acquire);                          \
        if (_owner != (cmd_worker_id)) {                                            \
            throw std::runtime_error(                                               \
                std::string(ctx) + ": worker " +                                   \
                std::to_string(cmd_worker_id) + " does not own slot " +            \
                std::to_string(slot_id) + " (owner=" + std::to_string(_owner) + ")");\
        }                                                                           \
    } while (0)

namespace py = pybind11;

// ===== SLOT-BASED DATA STRUCTURES =====

// Per-slot information (aggregator-side)
struct alignas(64) SlotInfo {
    std::atomic<int> owner_worker_id;  // -1 if free, worker_id otherwise

    void* input_buffer;                // Dynamically-sized input buffer
    void* output_buffer;               // Dynamically-sized output buffer

    cudaIpcMemHandle_t input_ipc_handle;
    cudaIpcMemHandle_t output_ipc_handle;

};

// Global slot pool
struct SlotPool {
    SlotInfo slots[MAX_SLOTS];
    AtomicSlotMask free_mask;             // Bit i = 1 means slot i is free
    AtomicSlotMask active_mask;           // Bit i = 1 means slot i is in barrier
    std::atomic<int> active_slot_count;   // Number of active slots
    int num_slots;                        // Number of slots initialized
    int device_index;                     // Device where buffers are allocated

    // Model dimensions (set once during initialization)
    int hidden_dim;
    int intermediate_dim;
    int qkv_dim;

    // Dynamically computed buffer sizes
    size_t input_buffer_size;
    size_t output_buffer_size;
    int max_seq_per_step;             // Max tokens per slot per step (PREFILL_CHUNK_SIZE)

    bool initialized = false;
};

static SlotPool g_slot_pool;

// Per-worker sampling state (set once per request, used by OP_LM_HEAD)
struct WorkerSamplingState {
    bool do_sample = false;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;
};
static std::map<int, WorkerSamplingState> g_worker_sampling_state;

struct LayerWeights {
    // Fused weights (cat at load time, originals freed — same total memory)
    torch::Tensor qkv_weight;        // [q_dim+k_dim+v_dim, hidden] — 1 matmul instead of 3
    torch::Tensor gate_up_weight;    // [gate_dim+up_dim, hidden]   — 1 matmul instead of 2

    torch::Tensor o_weight;           // [hidden, hidden]
    torch::Tensor o_bias;

    torch::Tensor down_weight;        // [hidden, intermediate]
    torch::Tensor down_bias;

    // Dimension info for splitting QKV output on worker side
    int64_t q_dim, k_dim, v_dim;
    int64_t gate_dim, up_dim;
};

static std::vector<LayerWeights> g_layer_weights;
static int g_num_layers = 0;
static bool g_weights_initialized = false;
static int g_aggregator_device = 0;

// Embed and LM head weights (stored separately for efficient access)
static torch::Tensor g_embed_weight;      // [vocab_size, hidden_dim] (detached at init)
static torch::Tensor g_lm_head_weight;    // [vocab_size, hidden_dim] (detached at init)
static torch::Tensor g_lm_head_weight_t;  // [hidden_dim, vocab_size] (pre-transposed)
static int g_vocab_size = 0;
static bool g_embed_lm_head_initialized = false;

// Pre-allocated output staging buffers (avoid per-op torch::empty for GEMM output)
// Input gather uses torch::cat which has a batched copy kernel — faster than manual copy
struct StagingBuffers {
    torch::Tensor out_qkv;         // (max_tokens, qkv_dim)
    torch::Tensor out_hidden;      // (max_tokens, hidden_dim) — output for O/down
    torch::Tensor out_gateup;      // (max_tokens, gate_dim+up_dim)
    int64_t max_tokens = 0;
    bool initialized = false;
};
static StagingBuffers g_staging;

// Command and ControlBlock structs are defined in common/ipc_types.h

static ControlBlock* g_ctrl = nullptr;
static Command* g_ring = nullptr;
static int g_capacity = 0;




// ===== SLOT POOL MANAGEMENT =====

py::dict init_slot_pool(int num_slots, int device_index, int hidden_dim, int intermediate_dim, int num_kv_heads, int head_dim, int max_seq_per_step, int dtype_size) {
    if (num_slots <= 0 || num_slots > MAX_SLOTS) {
        throw std::runtime_error("num_slots must be between 1 and " + std::to_string(MAX_SLOTS));
    }

    CUDA_RT_CHECK(cudaSetDevice(device_index));

    g_slot_pool.num_slots = num_slots;
    g_slot_pool.device_index = device_index;
    g_slot_pool.hidden_dim = hidden_dim;
    g_slot_pool.intermediate_dim = intermediate_dim;
    g_slot_pool.qkv_dim = hidden_dim + (num_kv_heads * head_dim * 2);  // Q + K + V

    g_slot_pool.max_seq_per_step = max_seq_per_step;

    // Compute buffer sizes dynamically from model dims + max tokens per step
    int qkv_dim = hidden_dim + (num_kv_heads * head_dim * 2);
    g_slot_pool.input_buffer_size = (size_t)max_seq_per_step * std::max(intermediate_dim, hidden_dim) * dtype_size;
    g_slot_pool.output_buffer_size = (size_t)max_seq_per_step * std::max(intermediate_dim * 2, qkv_dim) * dtype_size;

    SlotMask initial_free = SlotMask::all_below(num_slots);
    g_slot_pool.free_mask.store(initial_free, std::memory_order_relaxed);  // All free
    g_slot_pool.active_mask.store(SlotMask(), std::memory_order_relaxed);
    g_slot_pool.active_slot_count.store(0, std::memory_order_relaxed);

    // Per-slot allocation: each slot gets its own cudaMalloc + IPC handle
    // Workers can only access their own slot's memory (hardware-level isolation)
    for (int i = 0; i < num_slots; ++i) {
        auto& slot = g_slot_pool.slots[i];
        slot.owner_worker_id.store(-1, std::memory_order_relaxed);

        CUDA_RT_CHECK(cudaMalloc(&slot.input_buffer, g_slot_pool.input_buffer_size));
        CUDA_RT_CHECK(cudaIpcGetMemHandle(&slot.input_ipc_handle, slot.input_buffer));

        CUDA_RT_CHECK(cudaMalloc(&slot.output_buffer, g_slot_pool.output_buffer_size));
        CUDA_RT_CHECK(cudaIpcGetMemHandle(&slot.output_ipc_handle, slot.output_buffer));
    }

    g_slot_pool.initialized = true;

    size_t total_kb = num_slots * (g_slot_pool.input_buffer_size + g_slot_pool.output_buffer_size) / 1024;
    printf("[C++ Aggregator] Slot pool: %d slots, %zu KB input + %zu KB output per slot, %zu KB total (per-slot isolated)\n",
           num_slots, g_slot_pool.input_buffer_size / 1024, g_slot_pool.output_buffer_size / 1024, total_kb);

    py::dict result;
    result["num_slots"] = num_slots;
    result["input_buffer_size"] = g_slot_pool.input_buffer_size;
    result["output_buffer_size"] = g_slot_pool.output_buffer_size;
    return result;
}


[[nodiscard]] py::dict claim_slot(int worker_id) {
    if (!g_slot_pool.initialized) {
        throw std::runtime_error("claim_slot: slot pool not initialized");
    }

    // Atomically find and claim a free slot
    while (true) {
        SlotMask free = g_slot_pool.free_mask.load(std::memory_order_acquire);
        if (free.is_zero()) {
            throw std::runtime_error("No free slots available");
        }

        int slot_id = free.find_first();
        int word_idx = slot_id >> 6;
        uint64_t slot_bit = 1ULL << (slot_id & 63);

        // CAS on the specific word containing the found bit
        uint64_t expected_word = free.w[word_idx];
        if (g_slot_pool.free_mask.compare_exchange_word(
                word_idx, expected_word, expected_word & ~slot_bit,
                std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Successfully claimed
            auto& slot = g_slot_pool.slots[slot_id];
            slot.owner_worker_id.store(worker_id, std::memory_order_release);

            // Scrub any per-slot state a prior tenant may have left on this
            // slot_id so the new owner starts clean. A worker evicted on the
            // barrier timeout can set its ready bit AFTER it leaves active_mask
            // (so aggregator_signal_done_slots never clears it), and a crashed
            // worker can leave its evicted bit set. Without this, the next
            // tenant on this slot would either be skipped at the barrier
            // (stale ready bit -> GEMM on un-written input) or aborted on its
            // first token (stale evicted bit).
            if (g_ctrl) {
                SlotMask clear_bit = ~SlotMask::from_bit(slot_id);
                g_ctrl->per_slot_ready_mask.fetch_and_mask(clear_bit, std::memory_order_release);
                g_ctrl->evicted_slot_mask.fetch_and_mask(clear_bit, std::memory_order_release);
            }

            py::dict result;
            result["slot_id"] = slot_id;
            result["worker_id"] = worker_id;
            // Return per-slot handles — worker can only access this slot's memory
            result["input_handle"] = py::bytes(reinterpret_cast<char*>(&slot.input_ipc_handle),
                                               sizeof(cudaIpcMemHandle_t));
            result["output_handle"] = py::bytes(reinterpret_cast<char*>(&slot.output_ipc_handle),
                                                sizeof(cudaIpcMemHandle_t));
            result["input_buffer_size"] = g_slot_pool.input_buffer_size;
            result["output_buffer_size"] = g_slot_pool.output_buffer_size;

            printf("[C++ Aggregator] Slot %d claimed by worker %d\n", slot_id, worker_id);
            return result;
        }
        // CAS failed, retry
    }
}


void release_slot(int slot_id, int worker_id) {
    if (!g_slot_pool.initialized) {
        throw std::runtime_error("release_slot: slot pool not initialized");
    }
    if (slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
        throw std::runtime_error("release_slot: invalid slot_id");
    }

    auto& slot = g_slot_pool.slots[slot_id];

    // Verify ownership
    int expected = worker_id;
    if (!slot.owner_worker_id.compare_exchange_strong(expected, -1,
                                                       std::memory_order_acq_rel,
                                                       std::memory_order_acquire)) {
        throw std::runtime_error("release_slot: worker does not own slot " + std::to_string(slot_id));
    }

    // Zero slot buffers asynchronously before returning to the free pool.
    // Prevents the next tenant from replaying stale hidden states via
    // the input buffer without calling prepare_input_slot first.
    // cudaMemsetAsync: ~10µs GPU work, CPU returns immediately.
    if (slot.input_buffer) {
        cudaMemsetAsync(slot.input_buffer, 0, g_slot_pool.input_buffer_size,
                        cudaStreamPerThread);
    }
    if (slot.output_buffer) {
        cudaMemsetAsync(slot.output_buffer, 0, g_slot_pool.output_buffer_size,
                        cudaStreamPerThread);
    }
    // Wait for the zeroing to complete BEFORE the slot becomes claimable.
    // The next tenant maps these buffers and writes from a different process/
    // stream (via MPS) with no implicit ordering against this memset, so an
    // unsynchronized async zero could either land after the new tenant's first
    // write (clobbering it) or not finish before the new tenant reads (stale
    // residue). This is the cold release path, so the sync cost is acceptable.
    if (slot.input_buffer || slot.output_buffer) {
        cudaStreamSynchronize(cudaStreamPerThread);
    }

    // Return to free pool
    g_slot_pool.free_mask.fetch_or_bit(slot_id, std::memory_order_release);

    // Remove from active mask if present
    SlotMask clear = ~SlotMask::from_bit(slot_id);
    g_slot_pool.active_mask.fetch_and_mask(clear, std::memory_order_release);

    printf("[C++ Aggregator] Slot %d released by worker %d\n", slot_id, worker_id);
}


[[nodiscard]] int get_slot_owner(int slot_id) {
    if (!g_slot_pool.initialized || slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
        return -1;
    }
    return g_slot_pool.slots[slot_id].owner_worker_id.load(std::memory_order_acquire);
}


py::dict get_slot_pool_status() {
    py::dict result;
    result["initialized"] = g_slot_pool.initialized;
    result["num_slots"] = g_slot_pool.num_slots;
    SlotMask free_m = g_slot_pool.free_mask.load(std::memory_order_acquire);
    SlotMask active_m = g_slot_pool.active_mask.load(std::memory_order_acquire);
    result["free_mask_lo"] = free_m.w[0];
    result["free_mask_hi"] = free_m.w[1];
    result["active_mask_lo"] = active_m.w[0];
    result["active_mask_hi"] = active_m.w[1];
    result["free_count"] = free_m.popcount();
    result["active_count"] = active_m.popcount();
    result["active_slot_count"] = g_slot_pool.active_slot_count.load(std::memory_order_acquire);

    py::list slot_owners;
    for (int i = 0; i < g_slot_pool.num_slots; ++i) {
        slot_owners.append(g_slot_pool.slots[i].owner_worker_id.load(std::memory_order_acquire));
    }
    result["slot_owners"] = slot_owners;

    return result;
}


// Weight and queue initialization

// Layer-by-layer weight initialization (low peak memory)
// Call sequence: init_aggregator_begin → N × init_aggregator_layer → init_aggregator_finalize
// Python frees each layer's originals between init_aggregator_layer calls,
// so peak overhead is 1 layer's fused copy (~272 MiB) instead of all 32 (~8.5 GiB).

void init_aggregator_begin(int num_layers, int device_index) {
    g_num_layers = num_layers;
    g_layer_weights.resize(num_layers);
    g_aggregator_device = device_index;
    printf("[C++ Aggregator] Begin weight init: %d layers on device %d\n",
           num_layers, device_index);
    fflush(stdout);
}

void init_aggregator_layer(
    int layer_idx,
    torch::Tensor q_weight, torch::Tensor k_weight, torch::Tensor v_weight,
    torch::Tensor o_weight, torch::Tensor o_bias,
    torch::Tensor gate_weight, torch::Tensor up_weight,
    torch::Tensor down_weight, torch::Tensor down_bias
) {
    if (layer_idx < 0 || layer_idx >= g_num_layers) {
        throw std::runtime_error("init_aggregator_layer: layer_idx " + std::to_string(layer_idx) +
                                  " out of range [0, " + std::to_string(g_num_layers) + ")");
    }
    auto& lw = g_layer_weights[layer_idx];

    lw.q_dim = q_weight.size(0);
    lw.k_dim = k_weight.size(0);
    lw.v_dim = v_weight.size(0);
    lw.gate_dim = gate_weight.size(0);
    lw.up_dim = up_weight.size(0);

    // Fuse QKV: cat along dim 0 — 1 matmul instead of 3
    lw.qkv_weight = torch::cat({q_weight, k_weight, v_weight}, 0).detach();

    // Fuse gate/up: cat along dim 0 — 1 matmul instead of 2
    lw.gate_up_weight = torch::cat({gate_weight, up_weight}, 0).detach();

    lw.o_weight = o_weight.detach();
    lw.o_bias = o_bias;
    lw.down_weight = down_weight.detach();
    lw.down_bias = down_bias;
}

void init_aggregator_finalize() {
    g_weights_initialized = true;

    printf("[C++ Aggregator] Weights initialized with fused QKV (%lld) and gate/up (%lld) per layer\n",
           (long long)(g_layer_weights[0].q_dim + g_layer_weights[0].k_dim + g_layer_weights[0].v_dim),
           (long long)(g_layer_weights[0].gate_dim + g_layer_weights[0].up_dim));

    // Initialize staging buffers if slot pool is ready
    if (g_slot_pool.initialized) {
        CUDA_RT_CHECK(cudaSetDevice(g_aggregator_device));
        int64_t max_tokens = (int64_t)g_slot_pool.num_slots * g_slot_pool.max_seq_per_step;
        auto opts = torch::TensorOptions()
            .dtype(g_layer_weights[0].qkv_weight.scalar_type())
            .device(torch::Device(torch::kCUDA, g_aggregator_device));

        auto& lw0 = g_layer_weights[0];
        int64_t qkv_dim = lw0.q_dim + lw0.k_dim + lw0.v_dim;
        int64_t gateup_dim = lw0.gate_dim + lw0.up_dim;

        g_staging.out_qkv = torch::empty({max_tokens, qkv_dim}, opts);
        g_staging.out_hidden = torch::empty({max_tokens, g_slot_pool.hidden_dim}, opts);
        g_staging.out_gateup = torch::empty({max_tokens, gateup_dim}, opts);
        g_staging.max_tokens = max_tokens;
        g_staging.initialized = true;

        size_t staging_mb = max_tokens * (
            qkv_dim + g_slot_pool.hidden_dim + gateup_dim
        ) * 2 / (1024 * 1024);
        printf("[C++ Aggregator] Staging buffers: max_tokens=%lld, %zu MiB\n",
               (long long)max_tokens, staging_mb);
    }
    fflush(stdout);
}


void init_embed_lm_head_weights(
    torch::Tensor embed_weight,
    torch::Tensor lm_head_weight
) {
    g_embed_weight = embed_weight.detach();      // [vocab_size, hidden_dim]
    g_lm_head_weight = lm_head_weight.detach();  // [vocab_size, hidden_dim]
    g_lm_head_weight_t = g_lm_head_weight.t();   // [hidden_dim, vocab_size]
    g_vocab_size = static_cast<int>(embed_weight.size(0));
    g_embed_lm_head_initialized = true;
    printf("[C++ Aggregator] Embed/LM-head weights initialized: vocab=%d, hidden=%d\n",
           g_vocab_size, static_cast<int>(embed_weight.size(1)));
}


void init_aggregator_queue(
    const std::string& shm_name,
    int capacity
) {
    if (capacity <= 0) {
        throw std::runtime_error("init_aggregator_queue: capacity must be positive");
    }
    if (shm_name.empty()) {
        throw std::runtime_error("init_aggregator_queue: shm_name cannot be empty");
    }

    // Open existing shared memory created by ext_ipc_queue_events
    int fd = shm_open(shm_name.c_str(), O_RDWR, 0666);
    if (fd < 0) {
        throw std::runtime_error("Failed to open shared memory: " + shm_name);
    }

    // Calculate size
    size_t ctrl_bytes = sizeof(ControlBlock);
    size_t ring_bytes = static_cast<size_t>(capacity) * sizeof(Command);
    size_t shm_size = ctrl_bytes + ring_bytes;

    void* ptr = mmap(nullptr, shm_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) {
        throw std::runtime_error("Failed to mmap shared memory");
    }

    g_ctrl = reinterpret_cast<ControlBlock*>(ptr);
    g_ring = reinterpret_cast<Command*>(static_cast<char*>(ptr) + ctrl_bytes);
    g_capacity = capacity;
}


#include <thread>


// ===== SLOT-BASED GEMM EXECUTION =====

// Execute batched GEMM for all slots (fused weights + pre-allocated staging)
// QKV: 1 matmul instead of 3, gate/up: 1 matmul instead of 2
// Staging buffers eliminate per-op torch::empty + torch::cat allocations
void execute_batched_gemm_slots(
    const std::vector<Command*>& commands,
    int op, int layer,
    at::ScalarType dtype
) {
    if (commands.empty() || !g_slot_pool.initialized) return;

    int n = static_cast<int>(commands.size());

    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::Device(torch::kCUDA, g_aggregator_device));

    int64_t input_dim = (op == OP_DOWN_PROJ) ? g_slot_pool.intermediate_dim : g_slot_pool.hidden_dim;
    if (layer < 0 || layer >= g_num_layers) {
        throw std::runtime_error("execute_batched_gemm_slots: layer_idx " + std::to_string(layer) +
                                  " out of range [0, " + std::to_string(g_num_layers) + ")");
    }
    auto& lw = g_layer_weights[layer];

    // Single-slot fast path: direct IPC buffer access, no gather/scatter
    if (n == 1) {
        int slot_id = commands[0]->slot_id;
        if (slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
            throw std::runtime_error("execute_batched_gemm: invalid slot_id: " + std::to_string(slot_id));
        }
        VERIFY_SLOT_OWNER(slot_id, commands[0]->worker_id, "execute_batched_gemm N=1");
        int64_t bx0 = commands[0]->bx, tx0 = commands[0]->tx;
        if (bx0 <= 0 || tx0 <= 0 || bx0 > 65536 || tx0 > 65536) {
            throw std::runtime_error("execute_batched_gemm: invalid bx/tx dimensions");
        }
        void* input_ptr = g_slot_pool.slots[slot_id].input_buffer;
        void* output_ptr = g_slot_pool.slots[slot_id].output_buffer;
        int64_t slot_tokens = bx0 * tx0;
        auto x = torch::from_blob(input_ptr, {slot_tokens, input_dim}, options);

        if (op == OP_QKV_FUSED) {
            int64_t total_qkv = lw.q_dim + lw.k_dim + lw.v_dim;
            auto out = torch::from_blob(output_ptr, {slot_tokens, total_qkv}, options);
            torch::matmul_out(out, x, lw.qkv_weight.t());
        } else if (op == OP_O_PROJ) {
            auto out = torch::from_blob(output_ptr, {slot_tokens, g_slot_pool.hidden_dim}, options);
            torch::matmul_out(out, x, lw.o_weight.t());
            if (lw.o_bias.defined() && lw.o_bias.numel() > 0) out.add_(lw.o_bias);
        } else if (op == OP_GATE_UP_FUSED) {
            int64_t total_out = lw.gate_dim + lw.up_dim;
            auto out = torch::from_blob(output_ptr, {slot_tokens, total_out}, options);
            torch::matmul_out(out, x, lw.gate_up_weight.t());
        } else if (op == OP_DOWN_PROJ) {
            auto out = torch::from_blob(output_ptr, {slot_tokens, g_slot_pool.hidden_dim}, options);
            torch::matmul_out(out, x, lw.down_weight.t());
            if (lw.down_bias.defined() && lw.down_bias.numel() > 0) out.add_(lw.down_bias);
        }
        return;
    }

    // N>1: Flatten all slots to 2D, single GEMM, scatter back
    // Uses pre-allocated staging buffers to avoid per-op allocations

    // Compute per-slot token counts and total
    std::vector<int64_t> token_counts(n);
    int64_t total_tokens = 0;
    for (int i = 0; i < n; ++i) {
        int slot_id = commands[i]->slot_id;
        if (slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
            throw std::runtime_error("Invalid slot_id in command");
        }
        VERIFY_SLOT_OWNER(slot_id, commands[i]->worker_id, "execute_batched_gemm N>1");
        int64_t slot_tokens = commands[i]->bx * commands[i]->tx;
        token_counts[i] = slot_tokens;
        total_tokens += slot_tokens;
    }

    // Gather: create views of IPC slot buffers and cat into contiguous tensor
    // torch::cat uses a batched copy kernel (faster than N individual copy_ calls)
    std::vector<torch::Tensor> input_list;
    input_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        void* input_ptr = g_slot_pool.slots[commands[i]->slot_id].input_buffer;
        input_list.push_back(torch::from_blob(input_ptr, {token_counts[i], input_dim}, options));
    }
    torch::Tensor x_flat = torch::cat(input_list, 0);

    // GEMM + scatter (fused weights: 1 matmul per op instead of 2-3)
    if (op == OP_QKV_FUSED) {
        int64_t total_qkv = lw.q_dim + lw.k_dim + lw.v_dim;
        torch::Tensor output;
        if (g_staging.initialized && total_tokens <= g_staging.max_tokens) {
            output = g_staging.out_qkv.narrow(0, 0, total_tokens);
        } else {
            output = torch::empty({total_tokens, total_qkv}, options);
        }
        torch::matmul_out(output, x_flat, lw.qkv_weight.t());

        int64_t offset = 0;
        for (int i = 0; i < n; ++i) {
            void* output_ptr = g_slot_pool.slots[commands[i]->slot_id].output_buffer;
            torch::from_blob(output_ptr, {token_counts[i], total_qkv}, options)
                .copy_(output.narrow(0, offset, token_counts[i]));
            offset += token_counts[i];
        }
    }
    else if (op == OP_O_PROJ) {
        torch::Tensor output;
        if (g_staging.initialized && total_tokens <= g_staging.max_tokens) {
            output = g_staging.out_hidden.narrow(0, 0, total_tokens);
            torch::matmul_out(output, x_flat, lw.o_weight.t());
        } else {
            output = torch::matmul(x_flat, lw.o_weight.t());
        }
        if (lw.o_bias.defined() && lw.o_bias.numel() > 0) output.add_(lw.o_bias);

        int64_t offset = 0;
        for (int i = 0; i < n; ++i) {
            void* output_ptr = g_slot_pool.slots[commands[i]->slot_id].output_buffer;
            torch::from_blob(output_ptr, {token_counts[i], g_slot_pool.hidden_dim}, options)
                .copy_(output.narrow(0, offset, token_counts[i]));
            offset += token_counts[i];
        }
    }
    else if (op == OP_GATE_UP_FUSED) {
        int64_t total_out = lw.gate_dim + lw.up_dim;
        torch::Tensor output;
        if (g_staging.initialized && total_tokens <= g_staging.max_tokens) {
            output = g_staging.out_gateup.narrow(0, 0, total_tokens);
        } else {
            output = torch::empty({total_tokens, total_out}, options);
        }
        torch::matmul_out(output, x_flat, lw.gate_up_weight.t());

        int64_t offset = 0;
        for (int i = 0; i < n; ++i) {
            void* output_ptr = g_slot_pool.slots[commands[i]->slot_id].output_buffer;
            torch::from_blob(output_ptr, {token_counts[i], total_out}, options)
                .copy_(output.narrow(0, offset, token_counts[i]));
            offset += token_counts[i];
        }
    }
    else if (op == OP_DOWN_PROJ) {
        torch::Tensor output;
        if (g_staging.initialized && total_tokens <= g_staging.max_tokens) {
            output = g_staging.out_hidden.narrow(0, 0, total_tokens);
            torch::matmul_out(output, x_flat, lw.down_weight.t());
        } else {
            output = torch::matmul(x_flat, lw.down_weight.t());
        }
        if (lw.down_bias.defined() && lw.down_bias.numel() > 0) output.add_(lw.down_bias);

        int64_t offset = 0;
        for (int i = 0; i < n; ++i) {
            void* output_ptr = g_slot_pool.slots[commands[i]->slot_id].output_buffer;
            torch::from_blob(output_ptr, {token_counts[i], g_slot_pool.hidden_dim}, options)
                .copy_(output.narrow(0, offset, token_counts[i]));
            offset += token_counts[i];
        }
    }
}


// Execute batched embed using slot buffers (supports mixed prefill/decode shapes)
void execute_batched_embed_slots(
    const std::vector<Command*>& commands,
    at::ScalarType dtype
) {
    if (commands.empty() || !g_slot_pool.initialized || !g_embed_lm_head_initialized) return;

    int n = static_cast<int>(commands.size());

    auto token_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(torch::Device(torch::kCUDA, g_aggregator_device));

    auto output_opts = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::Device(torch::kCUDA, g_aggregator_device));

    int64_t vocab_size = g_embed_weight.size(0);

    // Single-slot fast path: skip cat + scatter
    if (n == 1) {
        int slot_id = commands[0]->slot_id;
        if (slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
            throw std::runtime_error("execute_batched_embed: invalid slot_id: " + std::to_string(slot_id));
        }
        VERIFY_SLOT_OWNER(slot_id, commands[0]->worker_id, "execute_batched_embed N=1");
        int64_t bx0 = commands[0]->bx, tx0 = commands[0]->tx;
        if (bx0 <= 0 || tx0 <= 0 || bx0 > 65536 || tx0 > 65536) {
            throw std::runtime_error("execute_batched_embed: invalid bx/tx dimensions");
        }
        int64_t slot_tokens = bx0 * tx0;
        auto tokens = torch::from_blob(g_slot_pool.slots[slot_id].input_buffer, {slot_tokens}, token_opts);
        tokens = tokens.clamp(0, vocab_size - 1);

        auto embeddings = torch::embedding(g_embed_weight, tokens).to(dtype);
        torch::from_blob(g_slot_pool.slots[slot_id].output_buffer,
                         {slot_tokens, g_slot_pool.hidden_dim}, output_opts).copy_(embeddings);
        return;
    }

    // Flatten: each slot (bx, tx) → (bx*tx,) tokens
    std::vector<torch::Tensor> token_list;
    std::vector<int64_t> token_counts(n);
    int64_t total_tokens = 0;
    token_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        int slot_id = commands[i]->slot_id;
        if (slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
            throw std::runtime_error("execute_batched_embed N>1: invalid slot_id: " + std::to_string(slot_id));
        }
        VERIFY_SLOT_OWNER(slot_id, commands[i]->worker_id, "execute_batched_embed N>1");
        int64_t bx = commands[i]->bx, tx = commands[i]->tx;
        if (bx <= 0 || tx <= 0 || bx > 65536 || tx > 65536) {
            throw std::runtime_error("execute_batched_embed N>1: invalid bx/tx dimensions");
        }
        void* input_ptr = g_slot_pool.slots[slot_id].input_buffer;
        int64_t slot_tokens = bx * tx;
        token_counts[i] = slot_tokens;
        total_tokens += slot_tokens;
        token_list.push_back(torch::from_blob(input_ptr, {slot_tokens}, token_opts));
    }
    torch::Tensor token_ids = torch::cat(token_list, 0);  // (total_tokens,)
    token_ids = token_ids.clamp(0, vocab_size - 1);

    // Embedding lookup
    torch::Tensor embeddings = torch::embedding(g_embed_weight, token_ids).to(dtype);
    // embeddings: (total_tokens, hidden_dim)

    // Scatter to slot output buffers
    int64_t offset = 0;
    for (int i = 0; i < n; ++i) {
        int slot_id = commands[i]->slot_id;
        void* output_ptr = g_slot_pool.slots[slot_id].output_buffer;
        torch::from_blob(output_ptr, {token_counts[i], g_slot_pool.hidden_dim}, output_opts)
            .copy_(embeddings.narrow(0, offset, token_counts[i]));
        offset += token_counts[i];
    }
}


// Execute batched LM head + sampling using slot buffers (supports mixed shapes)
void execute_batched_lm_head_sample_slots(
    const std::vector<Command*>& commands,
    at::ScalarType dtype
) {
    if (commands.empty() || !g_slot_pool.initialized || !g_embed_lm_head_initialized) return;

    int n = static_cast<int>(commands.size());

    // Get sampling params from stored per-worker state (set via OP_SET_SAMPLING)
    int worker_id = commands[0]->worker_id;
    if (worker_id < 0 || worker_id > 100000) {
        throw std::runtime_error("execute_batched_lm_head: invalid worker_id: " + std::to_string(worker_id));
    }
    auto it = g_worker_sampling_state.find(worker_id);
    bool do_sample = false;
    float temperature = 1.0f;
    int top_k = 0;
    float top_p = 1.0f;

    if (it != g_worker_sampling_state.end()) {
        do_sample = it->second.do_sample;
        temperature = it->second.temperature;
        top_k = it->second.top_k;
        top_p = it->second.top_p;
    }

    if (temperature < 0.01f) temperature = 0.01f;

    auto options = torch::TensorOptions()
        .dtype(dtype)
        .device(torch::Device(torch::kCUDA, g_aggregator_device));

    // Single-slot fast path: skip cat + scatter
    if (n == 1) {
        int slot_id = commands[0]->slot_id;
        if (slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
            throw std::runtime_error("execute_batched_lm_head: invalid slot_id: " + std::to_string(slot_id));
        }
        VERIFY_SLOT_OWNER(slot_id, commands[0]->worker_id, "execute_batched_lm_head N=1");
        int64_t bx0 = commands[0]->bx, tx0 = commands[0]->tx;
        if (bx0 <= 0 || tx0 <= 0 || bx0 > 65536 || tx0 > 65536) {
            throw std::runtime_error("execute_batched_lm_head: invalid bx/tx dimensions");
        }
        int64_t slot_tokens = bx0 * tx0;
        auto hidden = torch::from_blob(g_slot_pool.slots[slot_id].input_buffer,
                                        {slot_tokens, g_slot_pool.hidden_dim}, options);
        auto last_hidden = hidden.narrow(0, slot_tokens - 1, 1);  // last token
        auto logits = torch::matmul(last_hidden, g_lm_head_weight_t);

        torch::Tensor next_tokens;
        if (do_sample) {
            logits = logits / temperature;
            if (top_k > 0 && top_k < logits.size(-1)) {
                auto topk_result = torch::topk(logits, top_k, -1);
                auto topk_values = std::get<0>(topk_result);
                auto threshold = topk_values.index({torch::indexing::Slice(), -1}).unsqueeze(-1);
                logits = torch::where(logits < threshold,
                                      torch::full_like(logits, -std::numeric_limits<float>::infinity()),
                                      logits);
            }
            if (top_p < 1.0f) {
                auto sorted_result = torch::sort(logits, -1, true);
                auto sorted_logits = std::get<0>(sorted_result);
                auto sorted_indices = std::get<1>(sorted_result);
                auto probs = torch::softmax(sorted_logits, -1);
                auto cumulative_probs = torch::cumsum(probs, -1);
                auto sorted_indices_to_remove = cumulative_probs > top_p;
                sorted_indices_to_remove.index_put_({torch::indexing::Slice(), 0}, false);
                auto indices_to_remove = torch::zeros_like(logits, torch::kBool);
                indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove);
                logits = torch::where(indices_to_remove,
                                      torch::full_like(logits, -std::numeric_limits<float>::infinity()),
                                      logits);
            }
            auto probs = torch::softmax(logits, -1);
            next_tokens = torch::multinomial(probs, 1).squeeze(-1);
        } else {
            next_tokens = logits.argmax(-1);
        }

        auto token_opts = torch::TensorOptions()
            .dtype(torch::kInt64)
            .device(torch::Device(torch::kCUDA, g_aggregator_device));

        torch::from_blob(g_slot_pool.slots[slot_id].output_buffer, {1}, token_opts)
            .copy_(next_tokens);
        return;
    }

    // Gather last-token hidden states from each slot (for LM head we only need the last token)
    std::vector<torch::Tensor> last_hidden_list;
    last_hidden_list.reserve(n);
    for (int i = 0; i < n; ++i) {
        int slot_id = commands[i]->slot_id;
        if (slot_id < 0 || slot_id >= g_slot_pool.num_slots) {
            throw std::runtime_error("execute_batched_lm_head N>1: invalid slot_id: " + std::to_string(slot_id));
        }
        VERIFY_SLOT_OWNER(slot_id, commands[i]->worker_id, "execute_batched_lm_head N>1");
        int64_t bx = commands[i]->bx, tx = commands[i]->tx;
        if (bx <= 0 || tx <= 0 || bx > 65536 || tx > 65536) {
            throw std::runtime_error("execute_batched_lm_head N>1: invalid bx/tx dimensions");
        }
        void* input_ptr = g_slot_pool.slots[slot_id].input_buffer;
        int64_t slot_tokens = bx * tx;
        auto slot_hidden = torch::from_blob(input_ptr, {slot_tokens, g_slot_pool.hidden_dim}, options);
        // Take last token: row (slot_tokens - 1)
        last_hidden_list.push_back(slot_hidden.narrow(0, slot_tokens - 1, 1));
    }
    torch::Tensor last_hidden = torch::cat(last_hidden_list, 0);  // (n, hidden_dim)

    // LM head
    torch::Tensor logits = torch::matmul(last_hidden, g_lm_head_weight_t);

    torch::Tensor next_tokens;
    if (do_sample) {
        // Apply temperature
        logits = logits / temperature;

        // Apply top-k filtering
        if (top_k > 0 && top_k < logits.size(-1)) {
            auto topk_result = torch::topk(logits, top_k, -1);
            auto topk_values = std::get<0>(topk_result);
            auto threshold = topk_values.index({torch::indexing::Slice(), -1}).unsqueeze(-1);
            logits = torch::where(logits < threshold,
                                  torch::full_like(logits, -std::numeric_limits<float>::infinity()),
                                  logits);
        }

        // Apply top-p (nucleus) filtering
        if (top_p < 1.0f) {
            auto sorted_result = torch::sort(logits, -1, true);
            auto sorted_logits = std::get<0>(sorted_result);
            auto sorted_indices = std::get<1>(sorted_result);
            auto probs = torch::softmax(sorted_logits, -1);
            auto cumulative_probs = torch::cumsum(probs, -1);

            auto sorted_indices_to_remove = cumulative_probs > top_p;
            sorted_indices_to_remove.index_put_({torch::indexing::Slice(), 0}, false);

            auto indices_to_remove = torch::zeros_like(logits, torch::kBool);
            indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove);
            logits = torch::where(indices_to_remove,
                                  torch::full_like(logits, -std::numeric_limits<float>::infinity()),
                                  logits);
        }

        // Sample from distribution
        auto probs = torch::softmax(logits, -1);
        next_tokens = torch::multinomial(probs, 1).squeeze(-1);
    } else {
        // Greedy selection
        next_tokens = logits.argmax(-1);
    }

    auto token_opts = torch::TensorOptions()
        .dtype(torch::kInt64)
        .device(torch::Device(torch::kCUDA, g_aggregator_device));

    // Scatter to slot output buffers (one token ID per slot)
    for (int i = 0; i < n; ++i) {
        int slot_id = commands[i]->slot_id;
        void* output_ptr = g_slot_pool.slots[slot_id].output_buffer;
        torch::from_blob(output_ptr, {1}, token_opts)
            .copy_(next_tokens.narrow(0, i, 1));
    }
}


// Wait for active slots with leave detection
inline int aggregator_wait_for_slots(int active_count, SlotMask& active_mask) {
    if (active_count == 0) return 0;

    auto start = std::chrono::steady_clock::now();
    constexpr auto PER_SLOT_TIMEOUT  = std::chrono::milliseconds(500); // evict stuck slot after 500ms
    // 500ms = 45x worst-case legitimate barrier (~11ms: attention+LoRA for 16-token prefill chunk).
    // Healthy workers freeze for at most 500ms (~1 TPOT cycle) when one tenant hangs.
    // Safe: no legitimate op takes >11ms per barrier, so 500ms gives 45x margin.
    constexpr auto LEAVE_CHECK_INTERVAL = std::chrono::milliseconds(10);
    auto last_leave_check = start;

    int expected_count = active_count;
    int spin_count = 0;

    // Completion condition: every bit in active_mask has been set in per_slot_ready_mask
    // (per-slot mask lets us identify exactly which slot is stuck on timeout)
    while (true) {
        SlotMask ready = g_ctrl->per_slot_ready_mask.load(std::memory_order_acquire);
        SlotMask missing = active_mask & ~ready;
        if (missing.is_zero()) break;  // all active slots have signalled

        if (g_ctrl->quit_flag.load(std::memory_order_acquire) != 0) {
            return -1;  // clean shutdown requested
        }

        spin_count++;
        if (spin_count < SPIN_TIGHT) {
            // Tight spin
        } else if (spin_count < SPIN_YIELD) {
            #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
            #endif
        } else {
            auto now = std::chrono::steady_clock::now();

            // Periodically check for graceful mid-op leaves
            if (now - last_leave_check > LEAVE_CHECK_INTERVAL) {
                last_leave_check = now;

                SlotMask leaves = g_ctrl->pending_slot_leaves.load(std::memory_order_acquire);
                SlotMask leaving = leaves & active_mask;
                if (!leaving.is_zero()) {
                    active_mask &= ~leaving;
                    expected_count = active_mask.popcount();

                    g_ctrl->pending_slot_leaves.fetch_and_mask(~leaving, std::memory_order_acq_rel);
                    g_ctrl->active_slot_mask.store(active_mask, std::memory_order_release);
                    g_ctrl->active_slot_count.store(expected_count, std::memory_order_release);

                    printf("[Aggregator] Mid-op slot leave detected: mask=0x%llx:%llx, now waiting for %d slots\n",
                           (unsigned long long)leaving.w[1], (unsigned long long)leaving.w[0], expected_count);
                    fflush(stdout);

                    if (expected_count == 0) return 0;
                }
            }

            // Per-slot timeout: identify and evict stuck slots instead of global quit
            if (spin_count % 100000 == 0) {
                if (now - start > PER_SLOT_TIMEOUT) {
                    SlotMask rdy   = g_ctrl->per_slot_ready_mask.load(std::memory_order_acquire);
                    SlotMask stuck = active_mask & ~rdy;
                    if (!stuck.is_zero()) {
                        // Evict only the stuck slots — healthy slots continue unaffected
                        active_mask &= ~stuck;
                        expected_count = active_mask.popcount();
                        g_ctrl->active_slot_mask.store(active_mask, std::memory_order_release);
                        g_ctrl->active_slot_count.store(expected_count, std::memory_order_release);
                        // Signal eviction to the stuck workers so they abort cleanly
                        // instead of reading stale output buffers after waking up.
                        SlotMask remaining = stuck;
                        while (!remaining.is_zero()) {
                            int sid = remaining.find_first();
                            remaining.clear_bit(sid);
                            g_ctrl->evicted_slot_mask.fetch_or_bit(sid, std::memory_order_release);
                        }
                        fprintf(stderr,
                            "[Aggregator] EVICTING stuck slot(s) after %ds timeout "
                            "(mask=0x%llx:%llx). %d slot(s) continue.\n",
                            (int)PER_SLOT_TIMEOUT.count(),
                            (unsigned long long)stuck.w[1], (unsigned long long)stuck.w[0],
                            expected_count);
                        fflush(stderr);
                        if (expected_count == 0) return 0;  // all gone — outer loop re-waits
                        // Reset timer: give remaining slots a fresh window
                        start = std::chrono::steady_clock::now();
                        spin_count = 0;
                    }
                }
            }

            #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
                __builtin_ia32_pause();
            #endif
        }
    }

    return expected_count;
}


inline void aggregator_signal_done_slots() {
    g_ctrl->slot_ready_count.store(0, std::memory_order_release);
    // Clear per-slot ready bits for all currently active slots (reset for next barrier)
    SlotMask active = g_ctrl->active_slot_mask.load(std::memory_order_acquire);
    g_ctrl->per_slot_ready_mask.fetch_and_mask(~active, std::memory_order_release);
    g_ctrl->barrier_done_epoch.fetch_add(1, std::memory_order_release);
}

// Apply pending sampling states for newly joined slots
// Optimized: Uses find_first to only iterate over set bits
inline void apply_pending_sampling_states(const SlotMask& join_mask) {
    // Check if any joining slots have pending sampling state
    SlotMask pending = g_ctrl->pending_sampling_mask.load(std::memory_order_acquire);
    SlotMask to_apply = pending & join_mask;

    if (to_apply.is_zero()) return;

    // Iterate only over set bits using find_first + clear_bit
    SlotMask remaining = to_apply;
    while (!remaining.is_zero()) {
        int slot_id = remaining.find_first();
        remaining.clear_bit(slot_id);

        // Get the worker_id for this slot
        int worker_id = g_slot_pool.slots[slot_id].owner_worker_id.load(std::memory_order_acquire);
        if (worker_id >= 0) {
            // Copy sampling state from shared memory to g_worker_sampling_state
            const PendingSamplingState& ps = g_ctrl->pending_sampling[slot_id];
            WorkerSamplingState state;
            state.do_sample = ps.do_sample;
            state.temperature = ps.temperature;
            state.top_k = ps.top_k;
            state.top_p = ps.top_p;
            g_worker_sampling_state[worker_id] = state;
        }
    }

    // Clear the pending mask for slots we processed
    g_ctrl->pending_sampling_mask.fetch_and_mask(~to_apply, std::memory_order_release);
}


// Slot-based main loop (slots can join/leave at token boundaries)
void aggregator_main_loop_slots(int num_slots, int dtype_code) {
    if (!g_ctrl || !g_ring) {
        throw std::runtime_error("Aggregator queue not initialized");
    }
    if (!g_weights_initialized) {
        throw std::runtime_error("Aggregator weights not initialized");
    }
    if (!g_slot_pool.initialized) {
        throw std::runtime_error("Slot pool not initialized");
    }

    CUDA_RT_CHECK(cudaSetDevice(g_aggregator_device));
    at::ScalarType dtype = static_cast<at::ScalarType>(dtype_code);

    SlotMask active_mask;
    int tokens_processed = 0;
    int total_ops_processed = 0;

    printf("[C++ Aggregator] Starting SLOT-BASED main loop on device %d (%d slots)\n",
           g_aggregator_device, num_slots);
    fflush(stdout);

    while (g_ctrl->quit_flag.load(std::memory_order_acquire) == 0) {
        // ========================================
        // PHASE 1: Process slot joins/leaves (between tokens)
        // ========================================

        // Process leaves FIRST - slots that finished their token
        // Load-before-exchange: avoids claiming exclusive cache line in steady state
        SlotMask leaves;
        if (!g_ctrl->pending_slot_leaves.load(std::memory_order_acquire).is_zero())
            leaves = g_ctrl->pending_slot_leaves.exchange_zero(std::memory_order_acq_rel);
        if (!leaves.is_zero()) {
            active_mask &= ~leaves;
            int count = active_mask.popcount();
            printf("[Aggregator] Slots LEFT: mask=0x%llx:%llx, active: %d slots\n",
                   (unsigned long long)leaves.w[1], (unsigned long long)leaves.w[0], count);
            fflush(stdout);
        }

        // Process joins - new slots wanting to start
        // Workers that arrive close together will naturally batch on the next token
        // Load-before-exchange: avoids claiming exclusive cache line in steady state
        SlotMask joins;
        if (!g_ctrl->pending_slot_joins.load(std::memory_order_acquire).is_zero())
            joins = g_ctrl->pending_slot_joins.exchange_zero(std::memory_order_acq_rel);
        if (!joins.is_zero()) {
            // Apply pending sampling states BEFORE adding to active mask
            // This ensures sampling state is set before the slot participates in any ops
            apply_pending_sampling_states(joins);

            active_mask |= joins;
            int count = active_mask.popcount();
            printf("[Aggregator] Slots JOINED: mask=0x%llx:%llx, active: %d slots\n",
                   (unsigned long long)joins.w[1], (unsigned long long)joins.w[0], count);
            fflush(stdout);
        }

        int N = active_mask.popcount();
        g_ctrl->active_slot_mask.store(active_mask, std::memory_order_release);
        g_ctrl->active_slot_count.store(N, std::memory_order_release);

        if (N == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ========================================
        // PHASE 2: Process one full token (all layers)
        // ========================================

        int ops_per_token = 1 + g_num_layers * 4 + 1;  // embed + 4*layers + lm_head

        // Pre-allocate command vector outside loop (reused 130 times per token)
        std::vector<Command*> commands;
        commands.reserve(N);

        for (int op_idx = 0; op_idx < ops_per_token; ++op_idx) {
            // ── WAIT: spin for all active slots to signal ready ──
            int actual_slots = aggregator_wait_for_slots(N, active_mask);

            if (actual_slots < 0) {
                // quit_flag set externally (clean shutdown) — honour it
                aggregator_signal_done_slots();
                break;
            }

            if (actual_slots == 0) {
                // All slots evicted or left — exit ops loop, return to join/leave phase
                N = 0;
                break;
            }

            if (actual_slots != N) {
                N = actual_slots;  // some slots evicted mid-wait
            }

            if (g_ctrl->quit_flag.load(std::memory_order_acquire) != 0) break;

            // ── FETCH: read commands, filtering any from evicted slots ──
            // After eviction, a slot's command may be in the ring buffer if it
            // submitted before hanging. Filter by active_mask to discard such
            // dangling entries and keep the ring buffer consistent.
            commands.clear();

            int tail = g_ctrl->tail.load(std::memory_order_acquire);
            constexpr int MAX_SPIN = 10000000;
            constexpr int MAX_DRAIN = MAX_SLOTS + 8;  // safety cap on drain iterations
            int drained = 0;

            while ((int)commands.size() < N && drained < MAX_DRAIN) {
                Command* cmd = &g_ring[tail];

                int spin_count = 0;
                while (cmd->ready.load(std::memory_order_acquire) != 1) {
                    if (g_ctrl->quit_flag.load(std::memory_order_acquire) != 0) break;
                    if (++spin_count > MAX_SPIN) {
                        fprintf(stderr, "[C++ Aggregator] ERROR: command not ready after eviction drain\n");
                        g_ctrl->quit_flag.store(1, std::memory_order_release);
                        break;
                    }
                }

                if (g_ctrl->quit_flag.load(std::memory_order_acquire) != 0) break;

                cmd->ready.store(0, std::memory_order_release);
                tail = (tail + 1) % g_capacity;
                drained++;

                // Keep only commands from non-evicted (still active) slots
                if (active_mask.test_bit(cmd->slot_id)) {
                    commands.push_back(cmd);
                }
                // Otherwise: evicted slot's dangling command — consumed and discarded
            }

            g_ctrl->tail.store(tail, std::memory_order_release);

            if (commands.empty() || g_ctrl->quit_flag.load(std::memory_order_acquire) != 0) {
                if (g_ctrl->quit_flag.load(std::memory_order_acquire) != 0) {
                    aggregator_signal_done_slots();
                }
                continue;
            }

            int op = commands[0]->op_type;
            int layer = commands[0]->layer_idx;

            // Handle OP_SET_SAMPLING
            if (op == OP_SET_SAMPLING) {
                for (auto* cmd : commands) {
                    WorkerSamplingState state;
                    state.do_sample = (cmd->x_key & 1) != 0;
                    state.temperature = static_cast<float>(cmd->x_key >> 1) / 10000.0f;
                    state.top_k = static_cast<int>(cmd->y0_key);
                    state.top_p = static_cast<float>(cmd->y1_key) / 10000.0f;
                    g_worker_sampling_state[cmd->worker_id] = state;
                }
                aggregator_signal_done_slots();
                continue;
            }

            // ── COMPUTE: execute batched operations (flattened, mixed shapes OK) ──
            if (op == OP_EMBED) {
                execute_batched_embed_slots(commands, dtype);
            } else if (op == OP_LM_HEAD) {
                execute_batched_lm_head_sample_slots(commands, dtype);
            } else {
                execute_batched_gemm_slots(commands, op, layer, dtype);
            }

            // ── SYNC: wait for GPU compute to finish ──
            CUDA_RT_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream()));

            // ── SIGNAL: tell workers results are ready ──
            aggregator_signal_done_slots();

            total_ops_processed++;
        }

        tokens_processed++;
    }

    printf("[C++ Aggregator] Slot-based loop exited after %d tokens (%d ops)\n",
           tokens_processed, total_ops_processed);
    fflush(stdout);
}


// ===== P2P WEIGHT TRANSFER (DONOR/RECEIVER AGGREGATOR) =====

// Export CUDA IPC handles for all weight tensors so another aggregator can import them via P2P
py::dict export_weight_ipc_handles() {
    if (!g_weights_initialized) {
        throw std::runtime_error("export_weight_ipc_handles: weights not initialized");
    }
    if (!g_embed_lm_head_initialized) {
        throw std::runtime_error("export_weight_ipc_handles: embed/lm_head not initialized");
    }

    py::dict result;
    result["num_layers"] = g_num_layers;
    result["device_index"] = g_aggregator_device;

    // Export per-layer fused weight handles
    py::list layer_handles;
    for (int i = 0; i < g_num_layers; ++i) {
        auto& lw = g_layer_weights[i];
        py::dict lh;

        // Helper: get IPC handle for a tensor
        auto get_handle = [](const torch::Tensor& t) -> py::bytes {
            cudaIpcMemHandle_t handle;
            CUDA_RT_CHECK(cudaIpcGetMemHandle(&handle, t.data_ptr()));
            return py::bytes(reinterpret_cast<char*>(&handle), sizeof(cudaIpcMemHandle_t));
        };

        lh["qkv_weight_handle"] = get_handle(lw.qkv_weight);
        lh["qkv_weight_shape"] = py::make_tuple(lw.qkv_weight.size(0), lw.qkv_weight.size(1));
        lh["o_weight_handle"] = get_handle(lw.o_weight);
        lh["o_weight_shape"] = py::make_tuple(lw.o_weight.size(0), lw.o_weight.size(1));
        lh["down_weight_handle"] = get_handle(lw.down_weight);
        lh["down_weight_shape"] = py::make_tuple(lw.down_weight.size(0), lw.down_weight.size(1));
        lh["gate_up_weight_handle"] = get_handle(lw.gate_up_weight);
        lh["gate_up_weight_shape"] = py::make_tuple(lw.gate_up_weight.size(0), lw.gate_up_weight.size(1));

        // Bias handles (if defined)
        lh["has_o_bias"] = lw.o_bias.defined() && lw.o_bias.numel() > 0;
        if (lw.o_bias.defined() && lw.o_bias.numel() > 0) {
            lh["o_bias_handle"] = get_handle(lw.o_bias);
            lh["o_bias_size"] = lw.o_bias.size(0);
        }
        lh["has_down_bias"] = lw.down_bias.defined() && lw.down_bias.numel() > 0;
        if (lw.down_bias.defined() && lw.down_bias.numel() > 0) {
            lh["down_bias_handle"] = get_handle(lw.down_bias);
            lh["down_bias_size"] = lw.down_bias.size(0);
        }

        // Dimension metadata
        lh["q_dim"] = lw.q_dim;
        lh["k_dim"] = lw.k_dim;
        lh["v_dim"] = lw.v_dim;
        lh["gate_dim"] = lw.gate_dim;
        lh["up_dim"] = lw.up_dim;

        layer_handles.append(lh);
    }
    result["layers"] = layer_handles;

    // Export embed and lm_head handles
    auto get_handle = [](const torch::Tensor& t) -> py::bytes {
        cudaIpcMemHandle_t handle;
        CUDA_RT_CHECK(cudaIpcGetMemHandle(&handle, t.data_ptr()));
        return py::bytes(reinterpret_cast<char*>(&handle), sizeof(cudaIpcMemHandle_t));
    };

    result["embed_weight_handle"] = get_handle(g_embed_weight);
    result["embed_weight_shape"] = py::make_tuple(g_embed_weight.size(0), g_embed_weight.size(1));
    result["lm_head_weight_handle"] = get_handle(g_lm_head_weight);
    result["lm_head_weight_shape"] = py::make_tuple(g_lm_head_weight.size(0), g_lm_head_weight.size(1));

    // dtype
    result["dtype_code"] = static_cast<int>(g_layer_weights[0].qkv_weight.scalar_type());

    printf("[C++ Aggregator] Exported IPC handles for %d layers + embed + lm_head\n", g_num_layers);
    fflush(stdout);
    return result;
}


// Import weights from another aggregator's IPC handles via P2P copy
void import_weights_from_ipc(py::dict handles_dict, int target_device_index) {
    CUDA_RT_CHECK(cudaSetDevice(target_device_index));
    g_aggregator_device = target_device_index;

    int num_layers = handles_dict["num_layers"].cast<int>();
    int dtype_code = handles_dict["dtype_code"].cast<int>();
    at::ScalarType dtype = static_cast<at::ScalarType>(dtype_code);
    auto opts = torch::TensorOptions().dtype(dtype).device(torch::Device(torch::kCUDA, target_device_index));

    g_num_layers = num_layers;
    g_layer_weights.resize(num_layers);

    py::list layer_handles = handles_dict["layers"];

    // Helper: open IPC handle, copy to local tensor, close handle
    auto ipc_copy = [&](py::bytes handle_bytes, std::vector<int64_t> shape) -> torch::Tensor {
        std::string handle_str = handle_bytes;
        cudaIpcMemHandle_t handle;
        memcpy(&handle, handle_str.data(), sizeof(cudaIpcMemHandle_t));

        void* remote_ptr = nullptr;
        CUDA_RT_CHECK(cudaIpcOpenMemHandle(&remote_ptr, handle, cudaIpcMemLazyEnablePeerAccess));

        // IPC maps the remote pointer into this process's address space.
        // Under MPS, the visible device is always cuda:0 regardless of
        // the physical GPU, so use the target device for the view.
        torch::Tensor remote_view = torch::from_blob(remote_ptr, shape, opts);
        torch::Tensor local = remote_view.clone();  // P2P copy to local GPU

        CUDA_RT_CHECK(cudaIpcCloseMemHandle(remote_ptr));
        return local;
    };

    auto ipc_copy_1d = [&](py::bytes handle_bytes, int64_t size) -> torch::Tensor {
        return ipc_copy(handle_bytes, {size});
    };

    printf("[C++ Aggregator] Importing weights from donor via P2P to device %d...\n", target_device_index);
    fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < num_layers; ++i) {
        py::dict lh = layer_handles[i].cast<py::dict>();
        auto& lw = g_layer_weights[i];

        // Dimension metadata
        lw.q_dim = lh["q_dim"].cast<int64_t>();
        lw.k_dim = lh["k_dim"].cast<int64_t>();
        lw.v_dim = lh["v_dim"].cast<int64_t>();
        lw.gate_dim = lh["gate_dim"].cast<int64_t>();
        lw.up_dim = lh["up_dim"].cast<int64_t>();

        // Fused weights
        auto qkv_shape = lh["qkv_weight_shape"].cast<py::tuple>();
        lw.qkv_weight = ipc_copy(lh["qkv_weight_handle"].cast<py::bytes>(),
                                  {qkv_shape[0].cast<int64_t>(), qkv_shape[1].cast<int64_t>()});

        auto o_shape = lh["o_weight_shape"].cast<py::tuple>();
        lw.o_weight = ipc_copy(lh["o_weight_handle"].cast<py::bytes>(),
                                {o_shape[0].cast<int64_t>(), o_shape[1].cast<int64_t>()});

        auto down_shape = lh["down_weight_shape"].cast<py::tuple>();
        lw.down_weight = ipc_copy(lh["down_weight_handle"].cast<py::bytes>(),
                                   {down_shape[0].cast<int64_t>(), down_shape[1].cast<int64_t>()});

        auto gateup_shape = lh["gate_up_weight_shape"].cast<py::tuple>();
        lw.gate_up_weight = ipc_copy(lh["gate_up_weight_handle"].cast<py::bytes>(),
                                      {gateup_shape[0].cast<int64_t>(), gateup_shape[1].cast<int64_t>()});

        // Biases
        if (lh["has_o_bias"].cast<bool>()) {
            lw.o_bias = ipc_copy_1d(lh["o_bias_handle"].cast<py::bytes>(), lh["o_bias_size"].cast<int64_t>());
        } else {
            lw.o_bias = torch::Tensor();
        }
        if (lh["has_down_bias"].cast<bool>()) {
            lw.down_bias = ipc_copy_1d(lh["down_bias_handle"].cast<py::bytes>(), lh["down_bias_size"].cast<int64_t>());
        } else {
            lw.down_bias = torch::Tensor();
        }

        if (i % 8 == 0 || i == num_layers - 1) {
            printf("[C++ Aggregator]   Layer %d/%d imported\n", i + 1, num_layers);
            fflush(stdout);
        }
    }

    // Import embed and lm_head
    auto embed_shape = handles_dict["embed_weight_shape"].cast<py::tuple>();
    g_embed_weight = ipc_copy(handles_dict["embed_weight_handle"].cast<py::bytes>(),
                               {embed_shape[0].cast<int64_t>(), embed_shape[1].cast<int64_t>()});

    auto lmh_shape = handles_dict["lm_head_weight_shape"].cast<py::tuple>();
    g_lm_head_weight = ipc_copy(handles_dict["lm_head_weight_handle"].cast<py::bytes>(),
                                  {lmh_shape[0].cast<int64_t>(), lmh_shape[1].cast<int64_t>()});
    g_lm_head_weight_t = g_lm_head_weight.t();
    g_vocab_size = static_cast<int>(g_embed_weight.size(0));
    g_embed_lm_head_initialized = true;

    // Finalize (staging buffers, g_weights_initialized)
    g_weights_initialized = true;
    init_aggregator_finalize();

    CUDA_RT_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[C++ Aggregator] P2P weight import complete: %d layers + embed + lm_head in %.1f ms\n",
           num_layers, elapsed_ms);
    fflush(stdout);
}


// Cleanup slot pool (free cudaMalloc'd buffers)
void cleanup_slot_pool() {
    if (!g_slot_pool.initialized) return;

    CUDA_RT_CHECK(cudaSetDevice(g_slot_pool.device_index));
    CUDA_RT_CHECK(cudaDeviceSynchronize());

    // Free per-slot allocations
    for (int i = 0; i < g_slot_pool.num_slots; ++i) {
        if (g_slot_pool.slots[i].input_buffer) {
            cudaFree(g_slot_pool.slots[i].input_buffer);
            g_slot_pool.slots[i].input_buffer = nullptr;
        }
        if (g_slot_pool.slots[i].output_buffer) {
            cudaFree(g_slot_pool.slots[i].output_buffer);
            g_slot_pool.slots[i].output_buffer = nullptr;
        }
    }

    g_slot_pool.initialized = false;
    printf("[C++ Aggregator] Slot pool cleaned up\n");
    fflush(stdout);
}


// Python module bindings

PYBIND11_MODULE(ext_aggregator, m) {
    m.doc() = "C++ Batched GEMM Aggregator for Multi-GPU LLM Inference";

    m.def("init_aggregator_begin", &init_aggregator_begin,
          py::arg("num_layers"), py::arg("device_index"),
          "Begin layer-by-layer weight initialization");

    m.def("init_aggregator_layer", &init_aggregator_layer,
          py::arg("layer_idx"),
          py::arg("q_weight"), py::arg("k_weight"), py::arg("v_weight"),
          py::arg("o_weight"), py::arg("o_bias"),
          py::arg("gate_weight"), py::arg("up_weight"),
          py::arg("down_weight"), py::arg("down_bias"),
          "Initialize one layer's weights (fuses QKV and gate/up)");

    m.def("init_aggregator_finalize", &init_aggregator_finalize,
          "Finalize weight init and create staging buffers");

    m.def("init_aggregator_queue", &init_aggregator_queue,
          py::arg("shm_name"),
          py::arg("capacity"),
          "Initialize aggregator with queue shared memory");

    m.def("init_embed_lm_head_weights", &init_embed_lm_head_weights,
          py::arg("embed_weight"),
          py::arg("lm_head_weight"),
          "Initialize embed and lm_head weights for aggregator-side execution");

    // ===== SLOT-BASED FUNCTIONS =====
    m.def("init_slot_pool", &init_slot_pool,
          py::arg("num_slots"),
          py::arg("device_index"),
          py::arg("hidden_dim"),
          py::arg("intermediate_dim"),
          py::arg("num_kv_heads"),
          py::arg("head_dim"),
          py::arg("max_seq_per_step"),
          py::arg("dtype_size"),
          "Initialize slot pool with per-slot input/output buffers and model dimensions");

    m.def("claim_slot", &claim_slot,
          py::arg("worker_id"),
          "Atomically claim a free slot for a worker");

    m.def("release_slot", &release_slot,
          py::arg("slot_id"),
          py::arg("worker_id"),
          "Release a slot back to the pool");

    m.def("get_slot_owner", &get_slot_owner,
          py::arg("slot_id"),
          "Get the owner worker_id of a slot (-1 if free)");

    m.def("get_slot_pool_status", &get_slot_pool_status,
          "Get current status of the slot pool");

    m.def("cleanup_slot_pool", &cleanup_slot_pool,
          "Free all slot pool GPU buffers");

    m.def("aggregator_main_loop_slots", &aggregator_main_loop_slots,
          py::arg("num_slots"),
          py::arg("dtype_code"),
          py::call_guard<py::gil_scoped_release>(),
          "Run the slot-based aggregator loop (blocking, releases GIL)");

    // P2P weight transfer (donor/receiver aggregator)
    m.def("export_weight_ipc_handles", &export_weight_ipc_handles,
          "Export CUDA IPC handles for all weight tensors (donor side)");

    m.def("import_weights_from_ipc", &import_weights_from_ipc,
          py::arg("handles_dict"),
          py::arg("target_device_index"),
          "Import weights from another aggregator via P2P copy (receiver side)");

    // Export op type constants
    m.attr("OP_EMBED") = OP_EMBED;
    m.attr("OP_LM_HEAD") = OP_LM_HEAD;
    m.attr("OP_SET_SAMPLING") = OP_SET_SAMPLING;

    // Export slot constants
    m.attr("MAX_SLOTS") = MAX_SLOTS;
}
