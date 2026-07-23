// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <aie_api/aie.hpp>
#include <aie_api/utils.hpp>
#include <adf/intrinsics.h>
#include <stdint.h>

#include "qwen3_constants_06b.h"
#include "record_format.h"

namespace {

#ifndef __SIGN_SIGNED
#define __SIGN_SIGNED 1
#endif

constexpr int32_t kActSliceBf16 = 256;
constexpr int32_t kGroupSize = qwen3::kQ4GroupSize;
constexpr int32_t kGroupsPerChunk = kActSliceBf16 / kGroupSize;
constexpr int32_t kQ4NxScaleBf16 = qwen3::kMainRowsPerTile * kGroupsPerChunk;
constexpr int32_t kQ4NxScaleBytes = kQ4NxScaleBf16 * 2;
constexpr int32_t kQ4NxOffsetBytes = kQ4NxScaleBytes;
constexpr int32_t kQ4NxDataOffsetBytes = kQ4NxScaleBytes + kQ4NxOffsetBytes;
constexpr int32_t kQ4NxRowsPerLane = qwen3::kMainRowsPerTile / 2;
constexpr int32_t kQ4NxDataBytesPerLane =
    kActSliceBf16 * (kQ4NxRowsPerLane / 2);
constexpr int32_t kQ4NxGroupBytesPerLane =
    kGroupSize * (kQ4NxRowsPerLane / 2);
using Acc16 = aie::accum<accfloat, kQ4NxRowsPerLane>;

__attribute__((always_inline)) static inline Acc16 zero_accum16() {
  return aie::zeros<accfloat, kQ4NxRowsPerLane>();
}

__attribute__((always_inline)) static inline aie::vector<bfloat16, kQ4NxRowsPerLane * 4>
load_q4_dim_quad(const uint8_t *__restrict group_data, unsigned dim_quad) {
  const uint4 *__restrict q4_ptr = reinterpret_cast<const uint4 *>(
      group_data + dim_quad * kQ4NxRowsPerLane * 2);
  aie::vector<uint4, kQ4NxRowsPerLane * 4> q4 =
      aie::load_v<kQ4NxRowsPerLane * 4, aie_dm_resource::d>(q4_ptr);
  aie::vector<uint8, kQ4NxRowsPerLane * 4> q8 = aie::unpack(q4);
  return aie::to_float<bfloat16>(q8, 0);
}

__attribute__((always_inline)) static inline void mac_q4_dim_quad(
    Acc16 &acc,
    const aie::vector<bfloat16, kQ4NxRowsPerLane * 4> &q_bf16,
    const aie::vector<bfloat16, kQ4NxRowsPerLane> &scale_vec,
    const aie::vector<bfloat16, kQ4NxRowsPerLane> &offset_vec,
    const aie::vector<bfloat16, kGroupSize> &act_group,
    int32_t dim) {
  aie::vector<bfloat16, kQ4NxRowsPerLane> q0 =
      q_bf16.template extract<kQ4NxRowsPerLane>(0);
  Acc16 scaled0 = aie::mul(q0, scale_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> dequant0 =
      aie::add(scaled0.template to_vector<bfloat16>(), offset_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> act0 =
      aie::broadcast<bfloat16, kQ4NxRowsPerLane>(act_group.get(dim));
  acc = aie::mac(acc, dequant0, act0);

  aie::vector<bfloat16, kQ4NxRowsPerLane> q1 =
      q_bf16.template extract<kQ4NxRowsPerLane>(1);
  Acc16 scaled1 = aie::mul(q1, scale_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> dequant1 =
      aie::add(scaled1.template to_vector<bfloat16>(), offset_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> act1 =
      aie::broadcast<bfloat16, kQ4NxRowsPerLane>(act_group.get(dim + 1));
  acc = aie::mac(acc, dequant1, act1);

  aie::vector<bfloat16, kQ4NxRowsPerLane> q2 =
      q_bf16.template extract<kQ4NxRowsPerLane>(2);
  Acc16 scaled2 = aie::mul(q2, scale_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> dequant2 =
      aie::add(scaled2.template to_vector<bfloat16>(), offset_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> act2 =
      aie::broadcast<bfloat16, kQ4NxRowsPerLane>(act_group.get(dim + 2));
  acc = aie::mac(acc, dequant2, act2);

  aie::vector<bfloat16, kQ4NxRowsPerLane> q3 =
      q_bf16.template extract<kQ4NxRowsPerLane>(3);
  Acc16 scaled3 = aie::mul(q3, scale_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> dequant3 =
      aie::add(scaled3.template to_vector<bfloat16>(), offset_vec);
  aie::vector<bfloat16, kQ4NxRowsPerLane> act3 =
      aie::broadcast<bfloat16, kQ4NxRowsPerLane>(act_group.get(dim + 3));
  acc = aie::mac(acc, dequant3, act3);
}

__attribute__((always_inline)) static inline void accum_q4nx_group_lane(
    Acc16 &acc, const uint8_t *__restrict group_data,
    const bfloat16 *__restrict scale_group,
    const bfloat16 *__restrict offset_group,
    const aie::vector<bfloat16, kGroupSize> &act_group) {
  aie::vector<bfloat16, kQ4NxRowsPerLane> scale_vec =
      aie::load_v<kQ4NxRowsPerLane, aie_dm_resource::a>(scale_group);
  aie::vector<bfloat16, kQ4NxRowsPerLane> offset_vec =
      aie::load_v<kQ4NxRowsPerLane, aie_dm_resource::b>(offset_group);

  aie::vector<bfloat16, kQ4NxRowsPerLane * 4> q_bf16_current =
      load_q4_dim_quad(group_data, 0);
  aie::pipelined_loop<kGroupSize / 4 - 1>(
      kGroupSize / 4 - 1, [&](unsigned dim_quad) __aie_inline {
    aie::vector<bfloat16, kQ4NxRowsPerLane * 4> q_bf16_next =
        load_q4_dim_quad(group_data, dim_quad + 1);
    mac_q4_dim_quad(acc, q_bf16_current, scale_vec, offset_vec, act_group,
                    static_cast<int32_t>(dim_quad) * 4);
    q_bf16_current = q_bf16_next;
  });
  mac_q4_dim_quad(acc, q_bf16_current, scale_vec, offset_vec, act_group,
                  kGroupSize - 4);
}

__attribute__((always_inline)) static inline void
accum_q4nx_chunk(Acc16 &acc0, Acc16 &acc1,
                 const bfloat16 *__restrict packed_chunk,
                 const int32_t *__restrict activation_words) {
  const bfloat16 *__restrict scales = packed_chunk;
  const bfloat16 *__restrict offsets = reinterpret_cast<const bfloat16 *>(
      reinterpret_cast<const uint8_t *>(packed_chunk) + kQ4NxScaleBytes);
  const uint8_t *__restrict packed_data =
      reinterpret_cast<const uint8_t *>(packed_chunk) + kQ4NxDataOffsetBytes;
  const bfloat16 *__restrict activation =
      reinterpret_cast<const bfloat16 *>(activation_words);

  aie::pipelined_loop<kGroupsPerChunk>(
      kGroupsPerChunk, [&](unsigned qgroup) __aie_inline {
    aie::vector<bfloat16, kGroupSize> act_group =
        aie::load_v<kGroupSize, aie_dm_resource::c>(
            activation + qgroup * kGroupSize);
    const int32_t row_group = qgroup * qwen3::kMainRowsPerTile;
    const int32_t data_group = qgroup * kQ4NxGroupBytesPerLane;
    accum_q4nx_group_lane(acc0, packed_data + data_group, scales + row_group,
                          offsets + row_group, act_group);
    accum_q4nx_group_lane(acc1,
                          packed_data + kQ4NxDataBytesPerLane + data_group,
                          scales + row_group + kQ4NxRowsPerLane,
                          offsets + row_group + kQ4NxRowsPerLane, act_group);
  });
}

__attribute__((always_inline)) static inline void
emit_record_payload(Acc16 &acc0, Acc16 &acc1, int32_t *record) {
  bfloat16 *payload = reinterpret_cast<bfloat16 *>(record + 1);
  aie::store_v(payload, acc0.template to_vector<bfloat16>());
  aie::store_v(payload + kQ4NxRowsPerLane,
               acc1.template to_vector<bfloat16>());
}

__attribute__((always_inline)) static inline void
emit_record(Acc16 &acc0, Acc16 &acc1, int32_t *record, int32_t header) {
  record[0] = header;
  emit_record_payload(acc0, acc1, record);
}

#ifdef QWEN3_ENABLE_MAIN16_CYCLE_PROFILE
__attribute__((always_inline)) static inline void store_u64_words(
    int32_t *record, int32_t index, uint64_t value) {
  record[index] = static_cast<int32_t>(value & 0xffffffffULL);
  record[index + 1] = static_cast<int32_t>(value >> 32);
}

__attribute__((always_inline)) static inline void
emit_cycle_record(Acc16 &acc0, Acc16 &acc1, int32_t *record, int32_t block,
                  uint64_t total_cycles, uint64_t compute_cycles) {
  record[0] = qwen3::kQCompactPacketId;
  emit_record_payload(acc0, acc1, record);
  record[1] = block;
  store_u64_words(record, 2, total_cycles);
  store_u64_words(record, 4, compute_cycles);
  record[6] = qwen3::kQChunksPerRecord;
  record[7] = kActSliceBf16;
  record[8] = qwen3::kMainRowsPerTile;
}
#endif

template <int32_t Records, int32_t ChunksPerRecord>
// Keep phase bodies out of the dispatcher so Chess does not merge all phase
// schedules into one larger program-memory footprint.
__attribute__((noinline)) static void
run_projection_body(const bfloat16 *__restrict wt_ping,
                    const bfloat16 *__restrict wt_pong,
                    const int32_t *__restrict act_ping,
                    const int32_t *__restrict act_pong,
                    int32_t *__restrict record_ping,
                    int32_t *__restrict record_pong, int32_t record_header,
                    int32_t *record_toggle) {
  for (int32_t block = 0; block < Records; block++)
      chess_loop_range(Records, Records) {
    Acc16 acc0 = zero_accum16();
    Acc16 acc1 = zero_accum16();

    for (int32_t chunk = 0; chunk < ChunksPerRecord; chunk++)
        chess_loop_range(ChunksPerRecord, ChunksPerRecord) {
      acquire_greater_equal(qwen3::kMainActivationFullCoreLock, 1);
      acquire_greater_equal(qwen3::kMainWeightFullCoreLock, 1);

      const bfloat16 *__restrict wt = (chunk & 1) == 0 ? wt_ping : wt_pong;
      const int32_t *__restrict act = (chunk & 1) == 0 ? act_ping : act_pong;
      accum_q4nx_chunk(acc0, acc1, wt, act);

      release(qwen3::kMainActivationEmptyCoreLock, 1);
      release(qwen3::kMainWeightEmptyCoreLock, 1);
    }

    acquire_greater_equal(qwen3::kMainRecordEmptyCoreLock, 1);
    int32_t *record = ((*record_toggle) & 1) == 0 ? record_ping : record_pong;
    emit_record(acc0, acc1, record, record_header);
    *record_toggle += 1;
    release(qwen3::kMainRecordFullCoreLock, 1);
  }
}

__attribute__((always_inline)) static inline void
run_q_only_body(bfloat16 *wt_ping, bfloat16 *wt_pong, int32_t *act_ping,
                int32_t *act_pong, int32_t *record_ping,
                int32_t *record_pong, int32_t *record_toggle) {
  run_projection_body<qwen3::kQBodyRecords, qwen3::kQChunksPerRecord>(
      wt_ping, wt_pong, act_ping, act_pong, record_ping, record_pong,
      qwen3::kQCompactPacketId, record_toggle);
}

__attribute__((always_inline)) static inline void
run_qkv_body(bfloat16 *wt_ping, bfloat16 *wt_pong, int32_t *act_ping,
             int32_t *act_pong, int32_t *record_ping, int32_t *record_pong,
             int32_t *record_toggle) {
  constexpr int32_t kQkvBodyRecords =
      qwen3::kQBodyRecords + qwen3::kKvBodyRecords + qwen3::kKvBodyRecords;
  run_projection_body<kQkvBodyRecords, qwen3::kQChunksPerRecord>(
      wt_ping, wt_pong, act_ping, act_pong, record_ping, record_pong,
      qwen3::kQCompactPacketId, record_toggle);
}

__attribute__((always_inline)) static inline void
run_o_body(bfloat16 *wt_ping, bfloat16 *wt_pong, int32_t *act_ping,
           int32_t *act_pong, int32_t *record_ping, int32_t *record_pong,
           int32_t *record_toggle) {
  run_projection_body<qwen3::kOBodyRecords, qwen3::kOChunksPerRecord>(
      wt_ping, wt_pong, act_ping, act_pong, record_ping, record_pong,
      qwen3::kOCompactPacketId, record_toggle);
}

__attribute__((always_inline)) static inline void
run_upgate_body(bfloat16 *wt_ping, bfloat16 *wt_pong, int32_t *act_ping,
                int32_t *act_pong, int32_t *record_ping,
                int32_t *record_pong, int32_t *record_toggle) {
  run_projection_body<qwen3::kUpGateReplays, qwen3::kUpGateChunksPerReplay>(
      wt_ping, wt_pong, act_ping, act_pong, record_ping, record_pong,
      qwen3::kFfnCompactPacketId, record_toggle);
}

__attribute__((always_inline)) static inline void
run_down_body(bfloat16 *wt_ping, bfloat16 *wt_pong, int32_t *act_ping,
              int32_t *act_pong, int32_t *record_ping,
              int32_t *record_pong, int32_t *record_toggle) {
  run_projection_body<qwen3::kDownBodyRecords, qwen3::kDownChunksPerRecord>(
      wt_ping, wt_pong, act_ping, act_pong, record_ping, record_pong,
      qwen3::kDownCompactPacketId, record_toggle);
}

} // namespace

extern "C" {

void q4nx_main16_layer_scheduler(bfloat16 *wt_ping, bfloat16 *wt_pong,
                                 int32_t *act_ping, int32_t *act_pong,
                                 int32_t *record_ping, int32_t *record_pong,
                                 int32_t group, int32_t row, int32_t num_rows,
                                 int32_t phase_limit) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  (void)group;
  (void)row;
  (void)num_rows;
  int32_t record_toggle = 0;
  if (phase_limit == qwen3::kQPhase + 1) {
    run_q_only_body(wt_ping, wt_pong, act_ping, act_pong, record_ping,
                    record_pong, &record_toggle);
    return;
  }

  if (phase_limit >= qwen3::kMain16PhaseLimitQkv) {
    run_qkv_body(wt_ping, wt_pong, act_ping, act_pong, record_ping,
                 record_pong, &record_toggle);
  }
  if (phase_limit >= qwen3::kMain16PhaseLimitQkvo) {
    run_o_body(wt_ping, wt_pong, act_ping, act_pong, record_ping, record_pong,
               &record_toggle);
  }
  if (phase_limit >= qwen3::kMain16PhaseLimitUpGate) {
    run_upgate_body(wt_ping, wt_pong, act_ping, act_pong, record_ping,
                    record_pong, &record_toggle);
  }
  if (phase_limit >= qwen3::kMain16PhaseLimitFull) {
    run_down_body(wt_ping, wt_pong, act_ping, act_pong, record_ping,
                  record_pong, &record_toggle);
  }
}

#ifdef QWEN3_ENABLE_MAIN16_CYCLE_PROFILE
void q4nx_main16_cycle_profile_scheduler(bfloat16 *wt_ping,
                                         bfloat16 *wt_pong, int32_t *act_ping,
                                         int32_t *act_pong,
                                         int32_t *record_ping,
                                         int32_t *record_pong, int32_t group,
                                         int32_t row, int32_t num_rows,
                                         int32_t phase_limit) {
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  (void)group;
  (void)row;
  (void)num_rows;
  (void)phase_limit;
  int32_t record_toggle = 0;
  ::aie::tile tile = ::aie::tile::current();

  for (int32_t block = 0; block < qwen3::kQBodyRecords; block++)
      chess_loop_range(qwen3::kQBodyRecords, qwen3::kQBodyRecords) {
    Acc16 acc0 = zero_accum16();
    Acc16 acc1 = zero_accum16();
    uint64_t compute_cycles = 0;
    uint64_t total_start = tile.cycles();

    for (int32_t chunk = 0; chunk < qwen3::kQChunksPerRecord; chunk++)
        chess_loop_range(qwen3::kQChunksPerRecord,
                         qwen3::kQChunksPerRecord) {
      acquire_greater_equal(qwen3::kMainActivationFullCoreLock, 1);
      acquire_greater_equal(qwen3::kMainWeightFullCoreLock, 1);

      bfloat16 *wt = (chunk & 1) == 0 ? wt_ping : wt_pong;
      int32_t *act = (chunk & 1) == 0 ? act_ping : act_pong;
      uint64_t compute_start = tile.cycles();
      accum_q4nx_chunk(acc0, acc1, wt, act);
      uint64_t compute_end = tile.cycles();
      compute_cycles += compute_end - compute_start;

      release(qwen3::kMainActivationEmptyCoreLock, 1);
      release(qwen3::kMainWeightEmptyCoreLock, 1);
    }

    uint64_t total_end = tile.cycles();
    acquire_greater_equal(qwen3::kMainRecordEmptyCoreLock, 1);
    int32_t *record = (record_toggle & 1) == 0 ? record_ping : record_pong;
    emit_cycle_record(acc0, acc1, record, block, total_end - total_start,
                      compute_cycles);
    record_toggle += 1;
    release(qwen3::kMainRecordFullCoreLock, 1);
  }
}
#endif

} // extern "C"
