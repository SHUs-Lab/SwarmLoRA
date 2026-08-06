// Common types and constants shared between aggregator, worker, and IPC queue extensions
// This header reduces duplication and ensures struct layouts match across files.

#ifndef IPC_TYPES_H
#define IPC_TYPES_H

#include <atomic>
#include <cstdint>
#include <pthread.h>

// ============================================================================
// Constants
// ============================================================================

// Maximum slots in the slot pool (supports up to 128 concurrent requests)
constexpr int MAX_SLOTS = 128;

// Operation type codes for GEMM operations
constexpr int OP_O_PROJ = 3;          // Output projection after attention
constexpr int OP_DOWN_PROJ = 6;       // Down projection in MLP
constexpr int OP_QKV_FUSED = 9;       // Fused Q/K/V projection
constexpr int OP_GATE_UP_FUSED = 10;  // Fused gate/up projection in MLP
constexpr int OP_EMBED = 11;          // Embedding lookup
constexpr int OP_LM_HEAD = 12;        // LM head projection + sampling
constexpr int OP_SET_SAMPLING = 13;   // Set per-worker sampling parameters

// Spin-wait thresholds for progressive backoff
// These provide consistent behavior across all spin loops
constexpr int SPIN_TIGHT = 64;      // Tight spin iterations (no pause/yield)
constexpr int SPIN_PAUSE = 1024;    // Pause instructions before yielding
constexpr int SPIN_YIELD = 10000;   // Yield iterations before timeout checks

// Number of 64-bit words needed for MAX_SLOTS bitmask
constexpr int SLOT_WORDS = (MAX_SLOTS + 63) / 64;  // 2

// ============================================================================
// SlotMask: Non-atomic 128-bit bitmask for local variables
// ============================================================================
//
// Every atomic bitmask operation touches exactly one bit (one slot).
// Using uint64_t w[2] with per-word operations is safe — no cross-word
// atomic operations are needed.

struct SlotMask {
    uint64_t w[SLOT_WORDS];

    SlotMask() : w{0, 0} {}
    explicit SlotMask(uint64_t w0, uint64_t w1 = 0) : w{w0, w1} {}

    // Bit manipulation
    void set_bit(int i)   { w[i >> 6] |=  (1ULL << (i & 63)); }
    void clear_bit(int i) { w[i >> 6] &= ~(1ULL << (i & 63)); }
    bool test_bit(int i) const { return (w[i >> 6] >> (i & 63)) & 1; }

    int popcount() const {
        return __builtin_popcountll(w[0]) + __builtin_popcountll(w[1]);
    }

    bool is_zero() const { return w[0] == 0 && w[1] == 0; }

    // Find first set bit, returns -1 if none
    int find_first() const {
        if (w[0] != 0) return __builtin_ctzll(w[0]);
        if (w[1] != 0) return 64 + __builtin_ctzll(w[1]);
        return -1;
    }

    // Bitwise operators
    SlotMask operator&(const SlotMask& o) const { return SlotMask(w[0] & o.w[0], w[1] & o.w[1]); }
    SlotMask operator|(const SlotMask& o) const { return SlotMask(w[0] | o.w[0], w[1] | o.w[1]); }
    SlotMask operator~() const { return SlotMask(~w[0], ~w[1]); }
    SlotMask& operator&=(const SlotMask& o) { w[0] &= o.w[0]; w[1] &= o.w[1]; return *this; }
    SlotMask& operator|=(const SlotMask& o) { w[0] |= o.w[0]; w[1] |= o.w[1]; return *this; }

    // Static helpers
    static SlotMask from_bit(int i) {
        SlotMask m;
        m.set_bit(i);
        return m;
    }

    // All bits [0, n) set
    static SlotMask all_below(int n) {
        SlotMask m;
        if (n <= 0) return m;
        if (n >= 128) { m.w[0] = ~0ULL; m.w[1] = ~0ULL; return m; }
        if (n >= 64) {
            m.w[0] = ~0ULL;
            m.w[1] = (n == 64) ? 0ULL : ((1ULL << (n - 64)) - 1);
        } else {
            m.w[0] = (1ULL << n) - 1;
            m.w[1] = 0;
        }
        return m;
    }
};

// ============================================================================
// AtomicSlotMask: Atomic 128-bit bitmask for shared-memory fields
// ============================================================================
//
// Per-word atomics are safe because every operation touches exactly one bit.
// No cross-word CAS is ever needed.

struct AtomicSlotMask {
    std::atomic<uint64_t> w[SLOT_WORDS];

    // Load both words (non-atomic across words, but each word is atomic)
    SlotMask load(std::memory_order order = std::memory_order_seq_cst) const {
        return SlotMask(w[0].load(order), w[1].load(order));
    }

    // Store both words
    void store(const SlotMask& m, std::memory_order order = std::memory_order_seq_cst) {
        w[0].store(m.w[0], order);
        w[1].store(m.w[1], order);
    }

    // Exchange both words to zero, return old value
    SlotMask exchange_zero(std::memory_order order = std::memory_order_seq_cst) {
        return SlotMask(w[0].exchange(0, order), w[1].exchange(0, order));
    }

    // Atomically set one bit (fetch_or on the correct word)
    void fetch_or_bit(int i, std::memory_order order = std::memory_order_seq_cst) {
        w[i >> 6].fetch_or(1ULL << (i & 63), order);
    }

    // Atomically clear bits using a mask (AND with mask on both words)
    void fetch_and_mask(const SlotMask& mask, std::memory_order order = std::memory_order_seq_cst) {
        w[0].fetch_and(mask.w[0], order);
        w[1].fetch_and(mask.w[1], order);
    }

    // Test one bit via atomic load
    bool test_bit(int i, std::memory_order order = std::memory_order_seq_cst) const {
        return (w[i >> 6].load(order) >> (i & 63)) & 1;
    }

    // CAS on a specific word (used by claim_slot)
    bool compare_exchange_word(int word_idx, uint64_t& expected, uint64_t desired,
                                std::memory_order success = std::memory_order_seq_cst,
                                std::memory_order failure = std::memory_order_seq_cst) {
        return w[word_idx].compare_exchange_weak(expected, desired, success, failure);
    }
};

// ============================================================================
// Shared Structures (MUST remain in sync across all extensions!)
// ============================================================================

// Sampling parameters written out-of-band before signal_join
struct PendingSamplingState {
    bool do_sample;
    float temperature;
    int top_k;
    float top_p;
};

// Command entry in the ring buffer (64-byte aligned for cache efficiency)
// Field order is critical - shared memory layout must match exactly
struct alignas(64) Command {
    std::atomic<int> ready;      // 0=empty, 1=data written and ready
    int op_type;                 // Operation type (OP_* constants)
    int layer_idx;               // Transformer layer index
    int slot_id;                 // Which slot this command uses
    int worker_id;               // Which worker submitted (for logging)
    int64_t bx, tx, dx;          // Shape [B, T, D]
    uint64_t x_key;              // Input slot key
    uint64_t y0_key;             // Output 0 slot key
    uint64_t y1_key;             // Output 1 slot key (for fused ops)
    uint64_t y2_key;             // Output 2 slot key (for QKV fused)
    int ticket_id;               // Ticket ID for compatibility
    char padding[4];             // Pad to 64-byte alignment
};

// Control block for slot-based batching (shared via POSIX shm)
// Field order is critical - shared memory layout must match exactly
// alignas(64) ensures sizeof(ControlBlock) is a multiple of 64,
// so the Command ring buffer that follows it in shared memory stays 64-byte aligned
struct alignas(64) ControlBlock {
    // Ring buffer pointers
    std::atomic<int> head;
    std::atomic<int> tail;
    int capacity;

    // Shutdown flag (atomic to prevent compiler caching in spin loops)
    std::atomic<int> quit_flag;

    // Worker/slot tracking
    int num_workers;
    int num_slots;

    // Reserved for ABI compatibility (was free_slot_mask, now in SlotPool)
    AtomicSlotMask _reserved_free_slot_mask;

    // Slot-based batching fields
    AtomicSlotMask active_slot_mask;      // Which slots are active in barrier
    AtomicSlotMask pending_slot_joins;    // Slots wanting to join
    AtomicSlotMask pending_slot_leaves;   // Slots wanting to leave
    AtomicSlotMask per_slot_ready_mask;   // Bit i set when slot i signals ready this barrier
    AtomicSlotMask evicted_slot_mask;     // Bit i set when aggregator evicts slot i (worker reads to abort)
    std::atomic<int> active_slot_count;          // Number of active slots
    std::atomic<int> slot_ready_count;           // Slots ready for current op (kept for worker wait_for_epoch)
    std::atomic<int64_t> barrier_done_epoch;      // Epoch counter for slot barrier (int64_t to prevent overflow)

    // Out-of-band sampling state
    AtomicSlotMask pending_sampling_mask;
    PendingSamplingState pending_sampling[MAX_SLOTS];

    char pad1[64];  // Avoid false sharing

    // Reserved for ABI compatibility (pure spin-wait design, no longer used)
    pthread_mutex_t _reserved_batch_mutex;
    pthread_cond_t _reserved_batch_ready_cond;
    pthread_cond_t _reserved_batch_done_cond;
};

// ============================================================================
// Utility Macros
// ============================================================================

#define CUDA_RT_CHECK(call) do {                                 \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        throw std::runtime_error(std::string("CUDA error: ") +   \
                                 cudaGetErrorString(_e));        \
    }                                                            \
} while (0)

#endif // IPC_TYPES_H
