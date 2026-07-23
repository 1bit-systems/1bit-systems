#pragma once

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "qwen3_constants.h"

namespace qwen3 {

static inline bfloat16 *record_payload_bf16(int32_t *record) {
    return reinterpret_cast<bfloat16 *>(record + 1);
}

static inline int32_t mylm_record_header_for_phase(int32_t phase) {
    if (phase == kQPhase || phase == kKPhase || phase == kVPhase) {
        return 0x1;
    }
    if (phase == kOPhase || phase == kDownPhase) {
        return 0x4;
    }
    if (phase == kUpPhase || phase == kGatePhase) {
        return 0x8;
    }
    return 0;
}

static inline int32_t compact_packet_id_for_phase(int32_t phase) {
    if (phase == kQPhase) {
        return kQCompactPacketId;
    }
    if (phase == kKPhase) {
        return kKCompactPacketId;
    }
    if (phase == kVPhase) {
        return kVCompactPacketId;
    }
    if (phase == kOPhase) {
        return kOCompactPacketId;
    }
    if (phase == kUpPhase || phase == kGatePhase) {
        return kFfnCompactPacketId;
    }
    if (phase == kDownPhase) {
        return kDownCompactPacketId;
    }
    return 0;
}

static inline int32_t projection_record_header(int32_t phase, int32_t group, int32_t row) {
    (void)group;
    (void)row;
    return mylm_record_header_for_phase(phase);
}

static inline int32_t body_record_header(int32_t phase, int32_t block, int32_t group, int32_t row) {
    (void)block;
    (void)group;
    (void)row;
    return mylm_record_header_for_phase(phase);
}

} // namespace qwen3
