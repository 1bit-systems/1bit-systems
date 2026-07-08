// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// bitnet_ternary_kernels.cc — Native 2-bit packed ternary matmul for Strix Halo NPU.
//
// True 4× memory density: decodes packed uint8 → ternary BF16 on-the-fly.
// Replaces main_projection_bitnet_fast.o for 2-bit weight format.
//
// Packed format: each uint8 = 4 ternary values:
//   bits[0:1]=v0, bits[2:3]=v1, bits[4:5]=v2, bits[6:7]=v3
//   mapping: 00→-1.0, 01→0.0, 10→+1.0, 11→-1.0

#include <aie_api/aie.hpp>
#include <stdint.h>

// ── Constants (self-contained — no bitnet_constants.h dependency) ─
constexpr int32_t kRowLanes = 32;          // 32 output rows per tile (M)
constexpr int32_t kTernaryPerByte = 4;     // 4 ternary values per packed byte
constexpr int32_t kInputPerVec = 32;       // 32 input dims processed per vector iteration
constexpr int32_t kChunkSize = 256;        // K dimension = 256 ternary per chunk
constexpr int32_t kChunkIterations = kChunkSize / kInputPerVec;  // 8
constexpr int32_t kPackedBytesPerChunk = kChunkSize / kTernaryPerByte; // 64 bytes
constexpr int32_t kScaleCount = kRowLanes; // 32 bf16 scales per chunk

// Phase limits (matching bitnet_constants.h)
constexpr int32_t kQPhase = 0;
constexpr int32_t kKPhase = 1;
constexpr int32_t kVPhase = 2;
constexpr int32_t kOPhase = 3;
constexpr int32_t kUpPhase = 4;
constexpr int32_t kGatePhase = 5;
constexpr int32_t kDownPhase = 6;
constexpr int32_t kPhaseLimitQkv = kVPhase + 1;
constexpr int32_t kPhaseLimitQkvo = kOPhase + 1;
constexpr int32_t kPhaseLimitUpGate = kGatePhase + 1;
constexpr int32_t kPhaseLimitFull = kDownPhase + 1;

// Record sizes
constexpr int32_t kQRecordsPerTile = 5;
constexpr int32_t kKRecordsPerTile = 2;
constexpr int32_t kVRecordsPerTile = 2;
constexpr int32_t kORecordsPerTile = 5;
constexpr int32_t kUpRecordsPerTile = 28;
constexpr int32_t kDownRecordsPerTile = 5;
constexpr int32_t kChunksPerRecord = 10;
constexpr int32_t kDownChunksPerRecord = 27;

// Record: 1 header word + 16 bf16 payload = 17 dwords
constexpr int32_t kRecordDwords = 17;
constexpr int32_t kRecordPayloadBf16 = (kRecordDwords - 1) * 2;  // 32 bf16

// Weight buffer layout per chunk: 64 bytes ternary + 64 bytes scales = 128 bytes
constexpr int32_t kWeightBytesPerChunk = kPackedBytesPerChunk + kScaleCount * 2;

// Packet IDs
constexpr int32_t kQPacketId = 0x1;
constexpr int32_t kKPacketId = 0x1;
constexpr int32_t kVPacketId = 0x1;
constexpr int32_t kOPacketId = 0x4;
constexpr int32_t kFfnPacketId = 0x8;
constexpr int32_t kDownPacketId = 0x4;

using Acc = aie::accum<accfloat, kRowLanes>;

// ── Decode 8 packed bytes → 32 BF16 ternary values ────────────
__attribute__((always_inline)) static inline aie::vector<bfloat16, kInputPerVec>
decode_ternary_8bytes(const uint8_t *__restrict packed) {
  aie::vector<bfloat16, kInputPerVec> result;

  for (int32_t i = 0; i < 8; i++) {
    uint8_t byte = packed[i];
    for (int32_t b = 0; b < 4; b++) {
      uint8_t code = (byte >> (b * 2)) & 3;
      bfloat16 val;
      if (code == 2)       val = bfloat16(1.0f);
      else if (code == 1)  val = bfloat16(0.0f);
      else                 val = bfloat16(-1.0f);
      result[i * 4 + b] = val;
    }
  }
  return result;
}

// ── Accumulate one chunk (K=256) for all 32 rows ──────────────
// acc[lane] += scale[lane] * sum_j(ternary[lane][j] * activation[j])
//
// Each output lane has its own ternary row and scale.
// All lanes share the same activation vector.
__attribute__((always_inline)) static inline void
accum_ternary_chunk(Acc &acc_total,
                    const uint8_t *__restrict weight_data,
                    const bfloat16 *__restrict scales,
                    const bfloat16 *__restrict activation) {
  // Load all 32 scales once
  aie::vector<bfloat16, kRowLanes> scale_vec = aie::load_v<kRowLanes>(scales);

  // 8 iterations, each: 8 bytes → 32 ternary values × 32 activation values
  for (int32_t iter = 0; iter < kChunkIterations; iter++) {
    int32_t byte_off = iter * 8;     // 0, 8, 16, ..., 56
    int32_t act_off  = iter * 32;    // 0, 32, 64, ..., 224

    aie::vector<bfloat16, kInputPerVec> tern = decode_ternary_8bytes(weight_data + byte_off);
    aie::vector<bfloat16, kInputPerVec> act  = aie::load_v<kInputPerVec>(activation + act_off);

    // tern * act → scalar (this is a dot product for each of 32 lanes)
    // We reduce the 32-element vector to a scalar, then broadcast × scale
    auto prod = aie::mul(tern, act);
    bfloat16 dot = aie::reduce_add(aie::to_vector<bfloat16>(prod));

    // Broadcast the dot product to all 32 lanes and MAC with scale
    aie::vector<bfloat16, kRowLanes> dot_vec = aie::broadcast<bfloat16, kRowLanes>(dot);
    acc_total = aie::mac(acc_total, dot_vec, scale_vec);
  }
}

// ── Emit one record ───────────────────────────────────────────
__attribute__((always_inline)) static inline void
emit_record(const Acc &acc, int32_t *record, int32_t header) {
  record[0] = header;
  bfloat16 *payload = reinterpret_cast<bfloat16 *>(record + 1);
  aie::vector<bfloat16, kRowLanes> result = acc.template to_vector<bfloat16>();
  aie::store_v(payload, result);
}

// ── Process one body (Records × ChunksPerRecord) ───────────────
template <int32_t Records, int32_t ChunksPerRec>
__attribute__((noinline)) static void
run_body(const uint8_t *__restrict wt_ping,
         const uint8_t *__restrict wt_pong,
         const int32_t *__restrict act_ping,
         const int32_t *__restrict act_pong,
         int32_t *__restrict rec_ping,
         int32_t *__restrict rec_pong,
         int32_t header,
         int32_t *toggle) {
  for (int32_t block = 0; block < Records; block++)
      chess_loop_range(Records, Records) {
    Acc acc = aie::zeros<accfloat, kRowLanes>();

    for (int32_t chunk = 0; chunk < ChunksPerRec; chunk++)
        chess_loop_range(ChunksPerRec, ChunksPerRec) {
      const uint8_t *wt = ((chunk & 1) == 0) ? wt_ping : wt_pong;
      const int32_t *act_src = ((chunk & 1) == 0) ? act_ping : act_pong;

      const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(
          wt + chunk * kWeightBytesPerChunk + kPackedBytesPerChunk);
      const bfloat16 *act_bf16 = reinterpret_cast<const bfloat16 *>(
          act_src + chunk * (kChunkSize / 2));  // i32 → bf16 offset

      accum_ternary_chunk(acc,
          wt + chunk * kWeightBytesPerChunk,
          scales,
          act_bf16);
    }

    int32_t *rec = ((*toggle & 1) == 0) ? rec_ping : rec_pong;
    emit_record(acc, rec, header);
    *toggle += 1;
  }
}

// ── Per-phase entry points ────────────────────────────────────

__attribute__((always_inline)) static inline void
run_q(const uint8_t *wp, const uint8_t *wq,
      const int32_t *ap, const int32_t *aq,
      int32_t *rp, int32_t *rq, int32_t *t) {
  run_body<kQRecordsPerTile, kChunksPerRecord>(wp, wq, ap, aq, rp, rq, kQPacketId, t);
}

__attribute__((always_inline)) static inline void
run_k(const uint8_t *wp, const uint8_t *wq,
      const int32_t *ap, const int32_t *aq,
      int32_t *rp, int32_t *rq, int32_t *t) {
  run_body<kKRecordsPerTile, kChunksPerRecord>(wp, wq, ap, aq, rp, rq, kKPacketId, t);
}

__attribute__((always_inline)) static inline void
run_v(const uint8_t *wp, const uint8_t *wq,
      const int32_t *ap, const int32_t *aq,
      int32_t *rp, int32_t *rq, int32_t *t) {
  run_body<kVRecordsPerTile, kChunksPerRecord>(wp, wq, ap, aq, rp, rq, kVPacketId, t);
}

__attribute__((always_inline)) static inline void
run_o(const uint8_t *wp, const uint8_t *wq,
      const int32_t *ap, const int32_t *aq,
      int32_t *rp, int32_t *rq, int32_t *t) {
  run_body<kORecordsPerTile, kChunksPerRecord>(wp, wq, ap, aq, rp, rq, kOPacketId, t);
}

__attribute__((always_inline)) static inline void
run_upgate(const uint8_t *wp, const uint8_t *wq,
           const int32_t *ap, const int32_t *aq,
           int32_t *rp, int32_t *rq, int32_t *t) {
  run_body<kUpRecordsPerTile, kChunksPerRecord>(wp, wq, ap, aq, rp, rq, kFfnPacketId, t);
}

__attribute__((always_inline)) static inline void
run_down(const uint8_t *wp, const uint8_t *wq,
          const int32_t *ap, const int32_t *aq,
          int32_t *rp, int32_t *rq, int32_t *t) {
  run_body<kDownRecordsPerTile, kDownChunksPerRecord>(wp, wq, ap, aq, rp, rq, kDownPacketId, t);
}

extern "C" {

// ── Main Scheduler — entry point for each main16 tile core ────
void bitnet_ternary_layer_scheduler(uint8_t *wt_ping, uint8_t *wt_pong,
                                    int32_t *act_ping, int32_t *act_pong,
                                    int32_t *rec_ping, int32_t *rec_pong,
                                    int32_t group, int32_t row,
                                    int32_t num_rows, int32_t phase_limit) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  (void)group; (void)row; (void)num_rows;

  int32_t toggle = 0;

  if (phase_limit >= kPhaseLimitQkv) {
    run_q(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
    run_k(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
    run_v(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
  } else {
    run_q(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
    return;
  }

  if (phase_limit >= kPhaseLimitQkvo) {
    run_o(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
  }
  if (phase_limit >= kPhaseLimitUpGate) {
    run_upgate(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
    run_upgate(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
  }
  if (phase_limit >= kPhaseLimitFull) {
    run_down(wt_ping, wt_pong, act_ping, act_pong, rec_ping, rec_pong, &toggle);
  }
}

// ── Standalone single-chunk microbenchmark ────────────────────
// Processes one K=256 chunk for 32 output rows.
// Input:  [64 bytes ternary][64 bytes scales] [256 bf16 activations]
// Output: [32 bf16] results
void bitnet_ternary_micro(const uint8_t *__restrict packed_input,
                          bfloat16 *__restrict output) {
  const bfloat16 *scales = reinterpret_cast<const bfloat16 *>(
      packed_input + kPackedBytesPerChunk);
  const bfloat16 *activations = reinterpret_cast<const bfloat16 *>(
      packed_input + kPackedBytesPerChunk + kScaleCount * 2);

  Acc acc = aie::zeros<accfloat, kRowLanes>();
  accum_ternary_chunk(acc, packed_input, scales, activations);

  aie::vector<bfloat16, kRowLanes> result = acc.template to_vector<bfloat16>();
  aie::store_v(output, result);
}

} // extern "C"
