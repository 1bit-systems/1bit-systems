#pragma once

#include <stdint.h>

namespace zr1 {

// Standard NPU tile constants (same for all models)
constexpr int32_t kMainRowsPerTile = 32;
constexpr int32_t kQ4KChunk = 256;
constexpr int32_t kQ4GroupSize = 32;
constexpr int32_t kRecordDwords = 17;
constexpr int32_t kRecordPayloadDwords = kRecordDwords - 1;
constexpr int32_t kRecordPayloadBf16 = kRecordPayloadDwords * 2;

// Phase indices
constexpr int32_t kQPhase = 0;
constexpr int32_t kKPhase = 1;
constexpr int32_t kVPhase = 2;
constexpr int32_t kOPhase = 3;
constexpr int32_t kUpPhase = 4;
constexpr int32_t kGatePhase = 5;
constexpr int32_t kDownPhase = 6;

constexpr int32_t kMain16PhaseLimitQkv = kVPhase + 1;
constexpr int32_t kMain16PhaseLimitQkvo = kOPhase + 1;
constexpr int32_t kMain16PhaseLimitUpGate = kGatePhase + 1;
constexpr int32_t kMain16PhaseLimitFull = kDownPhase + 1;

// Compact packet IDs
constexpr int32_t kQCompactPacketId = 0x1;
constexpr int32_t kKCompactPacketId = 0x1;
constexpr int32_t kVCompactPacketId = 0x1;
constexpr int32_t kOCompactPacketId = 0x4;
constexpr int32_t kFfnCompactPacketId = 0x8;
constexpr int32_t kDownCompactPacketId = 0x4;

// Lock IDs
constexpr int32_t kMainActivationEmptyLock = 0;
constexpr int32_t kMainActivationFullLock = 1;
constexpr int32_t kMainWeightEmptyLock = 2;
constexpr int32_t kMainWeightFullLock = 3;
constexpr int32_t kMainRecordEmptyLock = 4;
constexpr int32_t kMainRecordFullLock = 5;
constexpr int32_t kCoreLocalLockBase = 0x30;

// ===== ZR1-1.5B specific body/chunk constants =====
// Q:  H=1536 input, NH×HD=1536 output. 1536/512=3 body records, 1536/256=6 chunks/record
// K:  H=1536 input, NKV×HD=256 output. 256/512=1 body record, 1536/256=6 chunks/record
// V:  same as K
// O:  NH×HD=1536 input, H=1536 output. 1536/512=3 body records, 1536/256=6 chunks/record
// UP/GATE: H=1536 input, IM=8960 output. 8960/512=18 blocks total, 18 replays each → 36 total
// DOWN: IM=8960 input, H=1536 output. 1536/512=3 body records, 8960/256=35 chunks/record

constexpr int32_t kQBodyRecords = 3;           // 1536/512
constexpr int32_t kKvBodyRecords = 1;          // 256/512
constexpr int32_t kOBodyRecords = 3;           // 1536/512
constexpr int32_t kUpGateReplays = 36;         // (18+18) UP+GATE blocks
constexpr int32_t kDownBodyRecords = 3;        // 1536/512

constexpr int32_t kQChunksPerRecord = 6;       // 1536/256
constexpr int32_t kKvChunksPerRecord = 6;      // 1536/256
constexpr int32_t kOChunksPerRecord = 6;       // 1536/256
constexpr int32_t kUpGateChunksPerReplay = 6;  // 1536/256
constexpr int32_t kDownChunksPerRecord = 35;   // 8960/256

// Weight chunk base offsets
constexpr int32_t kQWeightChunkBase = 0;                          // 0
constexpr int32_t kKWeightChunkBase = 18;                         // 0 + 3×6
constexpr int32_t kVWeightChunkBase = 24;                         // 18 + 1×6
constexpr int32_t kFullLayerOWeightChunkBase = 30;                // 24 + 1×6
constexpr int32_t kFullLayerUpGateWeightChunkBase = 48;           // 30 + 3×6
constexpr int32_t kFullLayerDownWeightChunkBase = 264;            // 48 + 36×6
}
