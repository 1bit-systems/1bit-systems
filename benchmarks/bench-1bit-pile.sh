#!/usr/bin/env bash
# bench-1bit-pile.sh — sweep llama-bench across the ternary/sub-2-bit pile
# Output: /home/bcloud/claude output/bench-1bit-<DATE>.json (one JSON object per model)
set -uo pipefail

LLAMA_BENCH="${LLAMA_BENCH:-/home/bcloud/.cache/lemonade/bin/llamacpp/vulkan/llama-bench}"
PILE_ROOT="${PILE_ROOT:-/home/bcloud/halo-ai/models/ternary-test}"
OUT_DIR="${OUT_DIR:-/home/bcloud/claude output}"
TS="$(date +%Y%m%d-%H%M%S)"
JSON_OUT="${OUT_DIR}/bench-1bit-${TS}.json"
LOG_OUT="${OUT_DIR}/bench-1bit-${TS}.log"

mkdir -p "${OUT_DIR}"

# (label, gguf path)
ENTRIES=(
    "lily-bonsai-1.7B-IQ1_S|${PILE_ROOT}/lily-bonsai-1.7b-rq/Bonsai-1.7B-IQ1_S.gguf"
    "lily-bonsai-4B-IQ1_S|${PILE_ROOT}/lily-bonsai-4b-rq/Bonsai-4B-IQ1_S.gguf"
    "lily-bonsai-8B-IQ1_S|${PILE_ROOT}/lily-bonsai-8b-rq/Bonsai-8B-IQ1_S.gguf"
    "lily-bonsai-1.7B-Q2_K|${PILE_ROOT}/lily-bonsai-1.7b-rq/Bonsai-1.7B-Q2_K.gguf"
    "lily-bonsai-4B-Q2_K|${PILE_ROOT}/lily-bonsai-4b-rq/Bonsai-4B-Q2_K.gguf"
    "lily-bonsai-8B-Q2_K|${PILE_ROOT}/lily-bonsai-8b-rq/Bonsai-8B-Q2_K.gguf"
    "superkaiii-bonsai-4B-TQ1_0|${PILE_ROOT}/bonsai-4b-mainline/Ternary-Bonsai-4B-TQ1_0.gguf"
    "superkaiii-bonsai-4B-TQ2_0|${PILE_ROOT}/bonsai-4b-mainline/Ternary-Bonsai-4B-TQ2_0.gguf"
    "trilm-3.9B-TQ1_0|${PILE_ROOT}/trilm-3.9b/TriLM_3.9B_Unpacked-4.0B-TQ1_0.gguf"
    "trilm-3.9B-TQ2_0|${PILE_ROOT}/trilm-3.9b/TriLM_3.9B_Unpacked-4.0B-TQ2_0.gguf"
    "outlier-10B-V2-TQ1_0|${PILE_ROOT}/outlier-v2-10b/Outlier-10B-V2.TQ1_0.gguf"
    "outlier-10B-V2-TQ2_0|${PILE_ROOT}/outlier-v2-10b/Outlier-10B-V2.TQ2_0.gguf"
    "gianni-bitnet-3B-TQ2_0|${PILE_ROOT}/gianni-3b-tq2/bitnet_b1_58-3B-TQ2_0.gguf"
    "gianni-bitnet-large-TQ2_0|${PILE_ROOT}/gianni-large-tq2/bitnet_b1_58-large-TQ2_0.gguf"
    "tensorblock-smollm-135M-Q2_K|${PILE_ROOT}/tb-smollm-135m/Bitnet-SmolLM-135M-Q2_K.gguf"
)

echo "[" > "${JSON_OUT}"
FIRST=1
for entry in "${ENTRIES[@]}"; do
    LABEL="${entry%%|*}"
    GGUF="${entry##*|}"

    if [[ ! -f "${GGUF}" ]]; then
        echo "SKIP ${LABEL}: missing ${GGUF}" | tee -a "${LOG_OUT}"
        continue
    fi

    SIZE_MB="$(($(stat -c%s "${GGUF}") / (1024 * 1024)))"
    echo "=== ${LABEL} (${SIZE_MB} MB) ===" | tee -a "${LOG_OUT}"

    # llama-bench JSON output. -p 512 (prompt-eval), -n 128 (gen), -r 2 (reps).
    # -ngl 99 = full GPU offload (Vulkan).
    RAW="$("${LLAMA_BENCH}" \
        -m "${GGUF}" \
        -p 512 -n 128 -r 2 -ngl 99 \
        -o json 2>>"${LOG_OUT}" || true)"

    [[ -z "${RAW}" ]] && { echo "FAIL ${LABEL}: no output" | tee -a "${LOG_OUT}"; continue; }

    # Each invocation prints one JSON array. Tag with our label.
    if [[ ${FIRST} -eq 0 ]]; then echo "," >> "${JSON_OUT}"; fi
    FIRST=0
    {
        echo "{\"label\":\"${LABEL}\",\"gguf\":\"${GGUF}\",\"size_mb\":${SIZE_MB},\"results\":${RAW}}"
    } >> "${JSON_OUT}"

    echo "${RAW}" | tee -a "${LOG_OUT}"
done
echo "]" >> "${JSON_OUT}"

echo "DONE. JSON: ${JSON_OUT}  LOG: ${LOG_OUT}"
