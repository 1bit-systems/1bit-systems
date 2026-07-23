#pragma once

#include <stdint.h>

namespace qwen3 {

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

// Lock IDs (same for all models)
constexpr int32_t kMainActivationEmptyLock = 0;
constexpr int32_t kMainActivationFullLock = 1;
constexpr int32_t kMainWeightEmptyLock = 2;
constexpr int32_t kMainWeightFullLock = 3;
constexpr int32_t kMainRecordEmptyLock = 4;
constexpr int32_t kMainRecordFullLock = 5;
constexpr int32_t kCoreLocalLockBase = 0x30;
constexpr int32_t kMainActivationEmptyCoreLock = kCoreLocalLockBase + kMainActivationEmptyLock;
constexpr int32_t kMainActivationFullCoreLock = kCoreLocalLockBase + kMainActivationFullLock;
constexpr int32_t kMainWeightEmptyCoreLock = kCoreLocalLockBase + kMainWeightEmptyLock;
constexpr int32_t kMainWeightFullCoreLock = kCoreLocalLockBase + kMainWeightFullLock;
constexpr int32_t kMainRecordEmptyCoreLock = kCoreLocalLockBase + kMainRecordEmptyLock;
constexpr int32_t kMainRecordFullCoreLock = kCoreLocalLockBase + kMainRecordFullLock;

// ===== Qwen3-0.6B specific body/chunk constants =====
// Q:  H=1024 input, NH×HD=2048 output. 2048/512=4 body records, 1024/256=4 chunks/record
// K:  H=1024 input, NKV×HD=1024 output. 1024/512=2 body records, 1024/256=4 chunks/record
// V:  same as K
// O:  NH×HD=2048 input, H=1024 output. 1024/512=2 body records, 2048/256=8 chunks/record
// UP/GATE: H=1024 input, IM=3072 output. 3072/512=6 blocks total, 6 replays
// DOWN: IM=3072 input, H=1024 output. 1024/512=2 body records, 3072/256=12 chunks/record

constexpr int32_t kQBodyRecords = 4;          // 2048/512 (was 8 for 8B)
constexpr int32_t kKvBodyRecords = 2;         // 1024/512 (same as 8B)
constexpr int32_t kOBodyRecords = 2;          // 1024/512 (was 8 for 8B)
constexpr int32_t kUpGateReplays = 12;        // (6+6) UP+GATE blocks (was 48 for 8B)
constexpr int32_t kDownBodyRecords = 2;       // 1024/512 (was 8 for 8B)

constexpr int32_t kQChunksPerRecord = 4;      // 1024/256 (was 16 for 8B)
constexpr int32_t kKvChunksPerRecord = 4;     // 1024/256 (was 16 for 8B)
constexpr int32_t kOChunksPerRecord = 8;      // 2048/256 (was 16 for 8B)
constexpr int32_t kUpGateChunksPerReplay = 4; // 1024/256 (was 16 for 8B)
constexpr int32_t kDownChunksPerRecord = 12;  // 3072/256 (was 48 for 8B)

// Weight chunk base offsets (for weight streaming)
constexpr int32_t kQWeightChunkBase = 0;                              // 0
constexpr int32_t kKWeightChunkBase = 16;                             // 0 + 4×4
constexpr int32_t kVWeightChunkBase = 24;                             // 16 + 2×4
constexpr int32_t kFullLayerOWeightChunkBase = 32;                    // 24 + 2×4
constexpr int32_t kFullLayerUpGateWeightChunkBase = 48;               // 32 + 2×8
constexpr int32_t kFullLayerDownWeightChunkBase = 96;                 // 48 + 12×4
}
