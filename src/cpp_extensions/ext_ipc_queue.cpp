// IPC command queue + slot-based barrier for batched aggregator
//
// This extension creates a shared memory ring buffer for workers to submit
// commands to the aggregator, plus slot-based barrier synchronization.

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <stdexcept>
#include <cstring>
#include <string>
#include <sstream>
#include <chrono>
#include <atomic>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <ATen/cuda/CUDAContext.h>

// Shared types and constants
#include "common/ipc_types.h"

namespace py = pybind11;

#define CUDA_DRV_CHECK(x) do {                           \
    CUresult e = (x);                                    \
    if (e != CUDA_SUCCESS) {                             \
        const char* err = nullptr;                       \
        cuGetErrorString(e, &err);                       \
        throw std::runtime_error(err ? err               \
                                     : "CUDA driver error"); \
    }                                                    \
} while(0)

static void ensure_cuda_init() {
    static std::once_flag flag;
    std::call_once(flag, []{
        CUDA_DRV_CHECK(cuInit(0));
    });
}

// Global state
static void*        g_shm_base      = nullptr;
static size_t       g_shm_size      = 0;
static std::string  g_shm_name;
static int          g_shm_fd        = -1;

static Command*     g_ring          = nullptr;
static ControlBlock* g_ctrl         = nullptr;
static int          g_capacity      = 0;

static std::string make_shm_name() {
    pid_t pid = getpid();
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    auto ns  = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    std::ostringstream oss;
    oss << "/gpuqueue_" << pid << "_" << ns;
    return oss.str();
}

static void create_and_map_shm(const std::string& name, size_t size) {
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd == -1) {
        throw std::runtime_error("shm_open (create) failed");
    }
    if (ftruncate(fd, size) != 0) {
        ::close(fd);
        shm_unlink(name.c_str());
        throw std::runtime_error("ftruncate failed for shm");
    }

    void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        shm_unlink(name.c_str());
        throw std::runtime_error("mmap failed for shm");
    }

    g_shm_base = base;
    g_shm_size = size;
    g_shm_name = name;
    g_shm_fd   = fd;
}

static void open_and_map_shm(const std::string& name, size_t size) {
    int fd = shm_open(name.c_str(), O_RDWR, 0600);
    if (fd == -1) {
        throw std::runtime_error("shm_open (open) failed");
    }
    void* base = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed for shm (open)");
    }

    g_shm_base = base;
    g_shm_size = size;
    g_shm_name = name;
    g_shm_fd   = fd;
}

py::dict create_queue(int capacity, int max_tickets) {
    if (capacity <= 0) {
        throw std::runtime_error("capacity must be > 0");
    }
    if (max_tickets <= 0) {
        throw std::runtime_error("max_tickets must be > 0");
    }

    ensure_cuda_init();

    size_t ctrl_bytes = sizeof(ControlBlock);
    size_t ring_bytes = static_cast<size_t>(capacity) * sizeof(Command);
    size_t shm_size   = ctrl_bytes + ring_bytes;

    std::string shm_name = make_shm_name();
    create_and_map_shm(shm_name, shm_size);

    g_ctrl = reinterpret_cast<ControlBlock*>(g_shm_base);
    g_ring = reinterpret_cast<Command*>(
        reinterpret_cast<char*>(g_shm_base) + ctrl_bytes
    );
    g_capacity = capacity;

    g_ctrl->head.store(0, std::memory_order_relaxed);
    g_ctrl->tail.store(0, std::memory_order_relaxed);
    g_ctrl->capacity  = capacity;
    g_ctrl->quit_flag.store(0, std::memory_order_relaxed);
    g_ctrl->num_workers = 0;

    py::dict out;
    out["shm_name"]     = shm_name;
    out["capacity"]     = capacity;
    out["max_tickets"]  = max_tickets;
    return out;
}

py::dict open_queue(const std::string& shm_name,
                    int capacity,
                    int max_tickets)
{
    if (capacity <= 0) {
        throw std::runtime_error("capacity must be > 0");
    }
    if (max_tickets <= 0) {
        throw std::runtime_error("max_tickets must be > 0");
    }

    ensure_cuda_init();

    size_t ctrl_bytes = sizeof(ControlBlock);
    size_t ring_bytes = static_cast<size_t>(capacity) * sizeof(Command);
    size_t shm_size   = ctrl_bytes + ring_bytes;

    open_and_map_shm(shm_name, shm_size);

    g_ctrl = reinterpret_cast<ControlBlock*>(g_shm_base);
    g_ring = reinterpret_cast<Command*>(
        reinterpret_cast<char*>(g_shm_base) + ctrl_bytes
    );
    g_capacity = capacity;

    py::dict out;
    out["capacity"]    = capacity;
    out["max_tickets"] = max_tickets;
    return out;
}


// Initialize batch control for slot-based operation
void init_batch_control(int num_slots) {
    if (!g_ctrl) {
        throw std::runtime_error("init_batch_control: queue not initialized");
    }
    if (num_slots <= 0 || num_slots > MAX_SLOTS) {
        throw std::runtime_error("init_batch_control: num_slots must be between 1 and " +
                                  std::to_string(MAX_SLOTS));
    }

    g_ctrl->num_workers = num_slots;  // Track number of slots
    g_ctrl->num_slots = num_slots;

    // Reserved field (actual slot allocation uses g_slot_pool.free_mask in ext_aggregator.cpp)
    SlotMask zero;
    g_ctrl->_reserved_free_slot_mask.store(zero, std::memory_order_relaxed);

    // Initialize slot-based batching fields
    g_ctrl->active_slot_mask.store(zero, std::memory_order_relaxed);
    g_ctrl->pending_slot_joins.store(zero, std::memory_order_relaxed);
    g_ctrl->pending_slot_leaves.store(zero, std::memory_order_relaxed);
    g_ctrl->per_slot_ready_mask.store(zero, std::memory_order_relaxed);
    g_ctrl->evicted_slot_mask.store(zero, std::memory_order_relaxed);
    g_ctrl->active_slot_count.store(0, std::memory_order_relaxed);
    g_ctrl->slot_ready_count.store(0, std::memory_order_relaxed);
    g_ctrl->barrier_done_epoch.store(0, std::memory_order_relaxed);
    g_ctrl->pending_sampling_mask.store(zero, std::memory_order_relaxed);

    // Note: _reserved_batch_mutex and _reserved_batch_*_cond are unused
    // (pure spin-wait design). Kept in struct for layout compatibility.
}


// ===== SLOT-BASED BARRIER FUNCTIONS =====

// Slot signals intent to join at next token boundary
// Set pending sampling state for a slot (called BEFORE signal_join)
// This is out-of-band - not through the command queue
void slot_set_pending_sampling(int slot_id, bool do_sample, float temperature, int top_k, float top_p) {
    if (!g_ctrl) {
        throw std::runtime_error("slot_set_pending_sampling: queue not initialized");
    }
    if (slot_id < 0 || slot_id >= MAX_SLOTS) {
        throw std::runtime_error("slot_set_pending_sampling: slot_id out of range (0-" + std::to_string(MAX_SLOTS - 1) + ")");
    }

    // Write sampling state
    g_ctrl->pending_sampling[slot_id].do_sample = do_sample;
    g_ctrl->pending_sampling[slot_id].temperature = temperature;
    g_ctrl->pending_sampling[slot_id].top_k = top_k;
    g_ctrl->pending_sampling[slot_id].top_p = top_p;

    // Mark this slot as having pending sampling (with release to ensure writes are visible)
    g_ctrl->pending_sampling_mask.fetch_or_bit(slot_id, std::memory_order_release);
}

void slot_signal_join(int slot_id) {
    if (!g_ctrl) {
        throw std::runtime_error("slot_signal_join: queue not initialized");
    }
    if (slot_id < 0 || slot_id >= MAX_SLOTS) {
        throw std::runtime_error("slot_signal_join: slot_id out of range (0-" + std::to_string(MAX_SLOTS - 1) + ")");
    }
    g_ctrl->pending_slot_joins.fetch_or_bit(slot_id, std::memory_order_release);
}

// Slot signals intent to leave at next token boundary
void slot_signal_leave(int slot_id) {
    if (!g_ctrl) {
        throw std::runtime_error("slot_signal_leave: queue not initialized");
    }
    if (slot_id < 0 || slot_id >= MAX_SLOTS) {
        throw std::runtime_error("slot_signal_leave: slot_id out of range (0-" + std::to_string(MAX_SLOTS - 1) + ")");
    }
    g_ctrl->pending_slot_leaves.fetch_or_bit(slot_id, std::memory_order_release);
}

// Signal that a slot is ready for the current operation (increment ready count)
void slot_signal_ready(int slot_id) {
    if (!g_ctrl) {
        throw std::runtime_error("slot_signal_ready: queue not initialized");
    }
    if (slot_id < 0 || slot_id >= MAX_SLOTS) {
        throw std::runtime_error("slot_signal_ready: slot_id out of range (0-" + std::to_string(MAX_SLOTS - 1) + ")");
    }
    // Set per-slot bit (enables aggregator to identify exactly which slot is stuck)
    g_ctrl->per_slot_ready_mask.fetch_or_bit(slot_id, std::memory_order_release);
    // Also increment count (kept for backward compat with worker wait_for_epoch)
    g_ctrl->slot_ready_count.fetch_add(1, std::memory_order_release);
}

// Wait for the aggregator to signal completion of current epoch
// IMPORTANT: Capture expected_epoch BEFORE calling slot_signal_ready()
[[nodiscard]] int64_t wait_for_epoch(int64_t expected_epoch) {
    if (!g_ctrl) {
        throw std::runtime_error("wait_for_epoch: queue not initialized");
    }

    int spin_count = 0;
    while (g_ctrl->barrier_done_epoch.load(std::memory_order_acquire) <= expected_epoch) {
        // Check quit flag
        if (g_ctrl->quit_flag.load(std::memory_order_acquire) != 0) {
            throw std::runtime_error("wait_for_epoch: quit flag set");
        }

        spin_count++;
        if (spin_count < SPIN_TIGHT) {
            // Tight spin - do nothing
        } else if (spin_count < SPIN_PAUSE) {
            #if defined(__x86_64__) || defined(_M_X64)
                __builtin_ia32_pause();
            #endif
        } else if (spin_count < SPIN_YIELD) {
            std::this_thread::yield();
        } else {
            // Brief sleep after extensive spinning
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }

    return g_ctrl->barrier_done_epoch.load(std::memory_order_acquire);
}

// Check if slot is currently active
[[nodiscard]] bool slot_is_active(int slot_id) {
    if (!g_ctrl) {
        return false;
    }
    if (slot_id < 0 || slot_id >= MAX_SLOTS) {
        return false;
    }
    return g_ctrl->active_slot_mask.test_bit(slot_id, std::memory_order_acquire);
}

// Get current barrier done epoch
int64_t get_barrier_done_epoch() {
    if (!g_ctrl) {
        return 0;
    }
    return g_ctrl->barrier_done_epoch.load(std::memory_order_acquire);
}

// Check if the aggregator evicted this slot (worker should abort its current request)
[[nodiscard]] bool is_slot_evicted(int slot_id) {
    if (!g_ctrl || slot_id < 0 || slot_id >= MAX_SLOTS) {
        return false;
    }
    return g_ctrl->evicted_slot_mask.test_bit(slot_id, std::memory_order_acquire);
}

// Clear the eviction flag after the worker has acknowledged and cleaned up
void clear_evicted_slot(int slot_id) {
    if (!g_ctrl || slot_id < 0 || slot_id >= MAX_SLOTS) {
        return;
    }
    SlotMask clear = ~SlotMask::from_bit(slot_id);
    g_ctrl->evicted_slot_mask.fetch_and_mask(clear, std::memory_order_release);
}

// Submit command with slot_id (scalar shape args to avoid heap-allocated vector)
void submit_slot_command(int op_type,
                         int layer_idx,
                         int slot_id,
                         int worker_id,
                         int64_t bx, int64_t tx, int64_t dx,
                         uint64_t x_key,
                         uint64_t y0_key,
                         uint64_t y1_key,
                         uint64_t y2_key,
                         int ticket_id)
{
    if (!g_ctrl || !g_ring) {
        throw std::runtime_error("submit_slot_command: queue not initialized");
    }

    // Validate ALL fields BEFORE reserving a ring entry. If validation ran
    // after the compare_exchange below, a rejected command would already have
    // advanced `head`, leaving a slot whose `ready` flag is never set — the
    // aggregator's fetch loop would then spin on it forever and shut down,
    // taking every co-resident tenant with it.
    if (slot_id < 0 || slot_id >= MAX_SLOTS) {
        throw std::runtime_error("submit_slot_command: slot_id out of range: " + std::to_string(slot_id));
    }
    if (worker_id < 0 || worker_id > 100000) {
        throw std::runtime_error("submit_slot_command: invalid worker_id: " + std::to_string(worker_id));
    }
    if (bx <= 0 || tx <= 0 || bx > 65536 || tx > 65536) {
        throw std::runtime_error("submit_slot_command: invalid bx/tx: bx=" + std::to_string(bx) + " tx=" + std::to_string(tx));
    }
    if ((int64_t)bx * (int64_t)tx > (int64_t)65536 * 65536) {
        throw std::runtime_error("submit_slot_command: bx*tx overflow");
    }

    // Thread-safe slot allocation using compare-exchange
    int head, next;
    do {
        head = g_ctrl->head.load(std::memory_order_acquire);
        next = (head + 1) % g_capacity;

        // Check for full buffer
        if (next == g_ctrl->tail.load(std::memory_order_acquire)) {
            throw std::runtime_error("command ring buffer FULL");
        }
    } while (!g_ctrl->head.compare_exchange_weak(head, next,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire));

    Command& cmd = g_ring[head];
    cmd.op_type   = op_type;
    cmd.layer_idx = layer_idx;
    cmd.slot_id   = slot_id;
    cmd.worker_id = worker_id;
    cmd.bx        = bx;
    cmd.tx        = tx;
    cmd.dx        = dx;
    cmd.x_key     = x_key;
    cmd.y0_key    = y0_key;
    cmd.y1_key    = y1_key;
    cmd.y2_key    = y2_key;
    cmd.ticket_id = ticket_id;

    // Mark command as ready (with release fence)
    cmd.ready.store(1, std::memory_order_release);
}

void set_quit(int value) {
    if (!g_ctrl) {
        throw std::runtime_error("set_quit: queue not initialized");
    }
    g_ctrl->quit_flag.store(value, std::memory_order_release);
    // Spin-wait loops check quit_flag directly, no condvar wakeup needed
}


PYBIND11_MODULE(ext_ipc_queue, m) {
    m.doc() = "IPC command queue + slot-based barrier for batched aggregator";

    // Queue creation/opening
    m.def("create_queue", &create_queue,
          py::arg("capacity"), py::arg("max_tickets"),
          "Create shared memory command queue");

    m.def("open_queue", &open_queue,
          py::arg("shm_name"),
          py::arg("capacity"), py::arg("max_tickets"),
          "Open shared memory command queue");

    // Barrier initialization
    m.def("init_batch_control", &init_batch_control,
          py::arg("num_slots"),
          "Initialize barrier control for slot-based operation");

    // Quit flag
    m.def("set_quit", &set_quit, py::arg("value"),
          "Set quit flag to signal shutdown");

    // Slot-based barrier functions
    m.def("slot_set_pending_sampling", &slot_set_pending_sampling,
          py::arg("slot_id"),
          py::arg("do_sample"),
          py::arg("temperature"),
          py::arg("top_k"),
          py::arg("top_p"),
          "Set pending sampling state for slot (call BEFORE signal_join)");

    m.def("slot_signal_join", &slot_signal_join,
          py::arg("slot_id"),
          "Signal intent to join sync batch at next token boundary");

    m.def("slot_signal_leave", &slot_signal_leave,
          py::arg("slot_id"),
          "Signal intent to leave sync batch at next token boundary");

    m.def("slot_signal_ready", &slot_signal_ready,
          py::arg("slot_id"),
          "Signal that slot is ready for current op (increment ready count without waiting)");

    m.def("wait_for_epoch", &wait_for_epoch,
          py::arg("expected_epoch"),
          "Wait for aggregator to complete epoch > expected_epoch (blocking). "
          "IMPORTANT: Capture expected_epoch BEFORE calling slot_signal_ready()");

    m.def("slot_is_active", &slot_is_active,
          py::arg("slot_id"),
          "Check if slot is currently active in sync batch");

    m.def("get_barrier_done_epoch", &get_barrier_done_epoch,
          "Get current barrier done epoch counter");

    m.def("is_slot_evicted", &is_slot_evicted,
          py::arg("slot_id"),
          "True if aggregator evicted this slot due to timeout. Worker should abort request.");

    m.def("clear_evicted_slot", &clear_evicted_slot,
          py::arg("slot_id"),
          "Clear eviction flag after worker has acknowledged and cleaned up.");

    m.def("submit_slot_command", &submit_slot_command,
          py::arg("op_type"), py::arg("layer_idx"), py::arg("slot_id"), py::arg("worker_id"),
          py::arg("bx"), py::arg("tx"), py::arg("dx"),
          py::arg("x_key"), py::arg("y0_key"),
          py::arg("y1_key"), py::arg("y2_key"),
          py::arg("ticket_id"),
          "Submit command with slot_id and scalar shape to ring buffer");
}
