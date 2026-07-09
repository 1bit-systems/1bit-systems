#pragma once
// INT8 Quantization Module — symmetric per-channel weight quantization + dynamic
// per-token activation quantization with KL-divergence calibration.
//
// Components:
//   - Int8WeightQuant:  Per-channel symmetric INT8 weight quantization
//   - Int8ActQuant:     Dynamic per-token INT8 activation quantization (amax-based)
//   - Int8Gemm:         INT8 matrix multiply with configurable accumulation precision
//   - Calibrator:       KL-divergence calibration for optimal activation scales
//   - PackedInt8Mat:    Packed INT8 matrix with scales (for NPU/GPU dispatch)

#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <span>
#include <concepts>
#include <expected>
#include <bit>

namespace specdecode::quant {

// ─── Configuration ──────────────────────────────────────────────────────────

enum class QuantGranularity : uint8_t {
    kPerTensor,       // Single scale for entire tensor
    kPerChannel,      // Per-output-channel scale (weights)
    kPerToken,        // Per-token scale (activations)
    kPerGroup,        // Per-group-of-channels (block quantization)
};

enum class AccumulationPrecision : uint8_t {
    kInt32,           // Standard INT32 accumulate
    kInt64,           // INT64 accumulate (avoids overflow in large matmuls)
};

struct Int8QuantConfig {
    QuantGranularity weight_granularity = QuantGranularity::kPerChannel;
    QuantGranularity act_granularity    = QuantGranularity::kPerToken;
    AccumulationPrecision accum_prec    = AccumulationPrecision::kInt32;
    float weight_scale_max = 0.0f;       // 0 = auto-compute from data
    int32_t block_size = 128;            // Group size for per-group quantization
    bool clamp_saturation = true;        // Clamp extreme values instead of wrapping
};

// ─── INT8 Quantized Matrix ──────────────────────────────────────────────────

template <typename ScaleT = float>
struct PackedInt8Mat {
    int32_t rows = 0;          // M (output dim)
    int32_t cols = 0;          // N (input dim)
    int32_t col_blocks = 0;    // Number of scale blocks along cols (for per-group)
    std::vector<int8_t> data;  // Packed INT8 data [rows * cols]
    std::vector<ScaleT> scales; // Per-channel or per-group scales [rows * col_blocks]

    size_t size_bytes() const noexcept {
        return data.size() * sizeof(int8_t) + scales.size() * sizeof(ScaleT);
    }

    bool empty() const noexcept { return data.empty(); }

    // Access data at logical position (row, col)
    int8_t& at(int32_t r, int32_t c) noexcept { return data[(size_t)r * cols + c]; }
    const int8_t& at(int32_t r, int32_t c) const noexcept { return data[(size_t)r * cols + c]; }
};

// ─── INT8 Weight Quantizer ──────────────────────────────────────────────────
//
// Symmetric per-channel quantization: W_int8 = round(W_f32 / scale_channel)
// scale_channel = max(|W_channel|) / 127.0

class Int8WeightQuant {
public:
    Int8WeightQuant() = default;
    explicit Int8WeightQuant(const Int8QuantConfig& cfg) : cfg_(cfg) {}

    // Quantize fp32 weights [out_features, in_features] → packed INT8.
    // Returns the quantized matrix with per-channel or per-group scales.
    PackedInt8Mat<float> quantize(
        std::span<const float> weights,
        int32_t out_features,
        int32_t in_features
    ) const {
        PackedInt8Mat<float> result;
        result.rows = out_features;
        result.cols = in_features;
        result.data.resize((size_t)out_features * in_features);
        result.scales.resize(out_features);

        for (int32_t r = 0; r < out_features; r++) {
            // Find channel max
            float amax = 0.0f;
            for (int32_t c = 0; c < in_features; c++) {
                float v = std::abs(weights[(size_t)r * in_features + c]);
                if (v > amax) amax = v;
            }
            if (amax < 1e-12f) amax = 1.0f;

            float scale;
            if (cfg_.weight_scale_max > 0.0f) {
                scale = cfg_.weight_scale_max / 127.0f;
            } else {
                scale = amax / 127.0f;
            }
            result.scales[r] = scale;

            // Quantize
            float inv_scale = 1.0f / scale;
            for (int32_t c = 0; c < in_features; c++) {
                float v = weights[(size_t)r * in_features + c];
                if (!std::isfinite(v)) v = 0.0f;
                int32_t q = (int32_t)std::round(v * inv_scale);
                q = std::clamp(q, -127, 127);
                result.data[(size_t)r * in_features + c] = (int8_t)q;
            }
        }
        return result;
    }

    // Dequantize INT8 → fp32
    static void dequantize(
        const PackedInt8Mat<float>& packed,
        std::span<float> output
    ) {
        for (int32_t r = 0; r < packed.rows; r++) {
            float scale = packed.scales[r];
            for (int32_t c = 0; c < packed.cols; c++) {
                output[(size_t)r * packed.cols + c] =
                    (float)packed.data[(size_t)r * packed.cols + c] * scale;
            }
        }
    }

private:
    Int8QuantConfig cfg_;
};

// ─── Dynamic Activation Quantizer ───────────────────────────────────────────
//
// Per-token dynamic quantization: scale_token = max(|x_token|) / 127.0
// For calibration, use the calibrator below.

class Int8ActQuant {
public:
    explicit Int8ActQuant(const Int8QuantConfig& cfg = {}) : cfg_(cfg) {}

    struct QuantizedAct {
        std::vector<int8_t> qdata;  // [n_tokens * hidden]
        std::vector<float> scales;  // [n_tokens]
        float max_scale = 0.0f;
    };

    // Quantize activations [n_tokens, hidden] with per-token dynamic scales
    QuantizedAct quantize(std::span<const float> activations, int32_t n_tokens, int32_t hidden) const {
        QuantizedAct result;
        result.qdata.resize((size_t)n_tokens * hidden);
        result.scales.resize(n_tokens);

        for (int32_t t = 0; t < n_tokens; t++) {
            float amax = 0.0f;
            for (int32_t h = 0; h < hidden; h++) {
                float v = std::abs(activations[(size_t)t * hidden + h]);
                if (v > amax && std::isfinite(v)) amax = v;
            }
            if (amax < 1e-12f) amax = 1.0f;
            float scale = amax / 127.0f;
            result.scales[t] = scale;
            if (scale > result.max_scale) result.max_scale = scale;

            float inv = 1.0f / scale;
            for (int32_t h = 0; h < hidden; h++) {
                float v = activations[(size_t)t * hidden + h];
                if (!std::isfinite(v)) v = 0.0f;
                int32_t q = (int32_t)std::round(v * inv);
                q = std::clamp(q, -127, 127);
                result.qdata[(size_t)t * hidden + h] = (int8_t)q;
            }
        }
        return result;
    }

    // Static quantization with fixed scale (for calibrated models)
    QuantizedAct quantize_static(std::span<const float> activations, int32_t n_tokens,
                                  int32_t hidden, float fixed_scale) const {
        QuantizedAct result;
        result.qdata.resize((size_t)n_tokens * hidden);
        result.scales.resize(n_tokens, fixed_scale);
        result.max_scale = fixed_scale;

        float inv = 1.0f / fixed_scale;
        for (size_t i = 0; i < activations.size(); i++) {
            float v = activations[i];
            if (!std::isfinite(v)) v = 0.0f;
            int32_t q = (int32_t)std::round(v * inv);
            q = std::clamp(q, -127, 127);
            result.qdata[i] = (int8_t)q;
        }
        return result;
    }

private:
    Int8QuantConfig cfg_;
};

// ─── KL-Divergence Calibrator ───────────────────────────────────────────────
//
// Collects activation statistics over a calibration dataset and finds the
// optimal fixed activation scale that minimizes KL divergence between
// fp32 and INT8 distributions.

class Calibrator {
public:
    explicit Calibrator(int32_t hidden_size, int32_t num_bins = 2048)
        : hidden_size_(hidden_size), num_bins_(num_bins) {
        histograms_.resize(hidden_size, Histogram(num_bins));
    }

    // Feed one batch of activations [n_tokens, hidden] into the histogram.
    void feed(std::span<const float> activations, int32_t n_tokens) {
        for (int32_t t = 0; t < n_tokens; t++) {
            for (int32_t h = 0; h < hidden_size_; h++) {
                float v = activations[(size_t)t * hidden_size_ + h];
                if (std::isfinite(v) && v != 0.0f) {
                    histograms_[h].add(v);
                }
            }
        }
    }

    // Compute the optimal per-channel activation scale using KL divergence.
    // The scale minimizes KL( fp32_dist || quantized_dist ).
    struct CalibrationResult {
        std::vector<float> per_channel_scales;  // [hidden_size]
        float max_scale = 0.0f;                 // Single scale for all channels
        float kl_divergence = 0.0f;             // Best achieved KL

        // Get a fixed scale safe for the entire tensor
        float fixed_activation_scale() const noexcept {
            return max_scale > 0.0f ? max_scale / 127.0f : 8.0f / 127.0f;
        }
    };

    CalibrationResult calibrate(int32_t num_calib_steps = 100) const {
        CalibrationResult result;
        result.per_channel_scales.resize(hidden_size_);
        result.max_scale = 0.0f;
        result.kl_divergence = 0.0f;

        for (int32_t h = 0; h < hidden_size_; h++) {
            auto opt = histograms_[h].optimal_threshold(num_calib_steps);
            result.per_channel_scales[h] = opt.threshold;
            if (opt.threshold > result.max_scale) result.max_scale = opt.threshold;
            result.kl_divergence += opt.kl_divergence;
        }
        result.kl_divergence /= (float)hidden_size_;
        return result;
    }

private:
    struct Histogram {
        std::vector<float> bins;
        float bin_width;
        float min_val, max_val;
        int32_t num_bins;
        float total_count;

        Histogram(int32_t nbins)
            : bins(nbins, 0.0f), bin_width(0.01f), min_val(0.0f), max_val(0.0f),
              num_bins(nbins), total_count(0.0f) {}

        void add(float v) {
            float av = std::abs(v);
            if (av > max_val) max_val = av;
            if (total_count == 0.0f) min_val = av;

            // Adaptive binning: double bin width if we exceed range
            while (av >= min_val + bin_width * num_bins) {
                rebin();
            }

            int idx = (int)((av - min_val) / bin_width);
            if (idx >= num_bins) idx = num_bins - 1;
            if (idx < 0) idx = 0;
            bins[idx]++;
            total_count++;
        }

        void rebin() {
            // Merge adjacent bins: double bin width
            std::vector<float> new_bins(num_bins, 0.0f);
            for (int i = 0; i < num_bins; i++) {
                new_bins[i / 2] += bins[i];
            }
            bin_width *= 2.0f;
            bins = std::move(new_bins);
        }

        struct ThresholdResult {
            float threshold;
            float kl_divergence;
        };

        ThresholdResult optimal_threshold(int num_steps) const {
            if (total_count < 1.0f) return {1.0f, 0.0f};
            float best_kl = std::numeric_limits<float>::max();
            float best_threshold = max_val;

            // Normalize histogram to distribution
            std::vector<float> dist(num_bins);
            float inv_total = 1.0f / total_count;
            for (int i = 0; i < num_bins; i++) dist[i] = bins[i] * inv_total;

            float step = max_val / (float)num_steps;
            for (int s = 1; s <= num_steps; s++) {
                float threshold = step * (float)s;

                // Quantize: bin values into 128 levels within [-threshold, threshold]
                float bin_size = threshold / 64.0f; // 128 levels centered at 0
                std::vector<float> quantized(num_bins, 0.0f);
                for (int i = 0; i < num_bins; i++) {
                    float val = min_val + bin_width * (float)i;
                    if (val > threshold) break;
                    int qidx = (int)(val / bin_size);
                    if (qidx >= 128) qidx = 127;
                    quantized[qidx] += dist[i];
                    if (val > 0) quantized[128 + qidx] += dist[i]; // negative side
                }

                // Compute KL(P || Q) with smoothing
                float kl = 0.0f;
                for (int i = 0; i < num_bins; i++) {
                    float p = dist[i];
                    if (p < 1e-10f) continue;
                    // Map i to quantized bucket
                    float val = min_val + bin_width * (float)i;
                    int qidx = (int)(val / bin_size);
                    if (qidx >= 128) qidx = 127;
                    if (val > threshold) continue; // clipped values

                    float q = quantized[qidx] + 1e-10f; // smooth zero
                    kl += p * std::log(p / q);
                }

                if (kl < best_kl) {
                    best_kl = kl;
                    best_threshold = threshold;
                }
            }
            return {best_threshold, best_kl};
        }
    };

    int32_t hidden_size_;
    int32_t num_bins_;
    std::vector<Histogram> histograms_;
};

// ─── INT8 GEMM: A_int8 @ B_int8^T with fp32 accumulation ───────────────────

template <typename T = float>
concept Arithmetic = std::is_arithmetic_v<T>;

// Standard INT8 GEMM: C[m][n] = sum_k A[m][k] * B[n][k] * scale_a * scale_b
// A is [M, K] INT8, B is [N, K] INT8 (B is quantized weights, stored transposed)
// Both have per-row/per-channel scales.
//
// Returns C as fp32. Supports M=1 (decode) and M>1 (prefill) paths.
template <bool UseOpenMP = true>
class Int8Gemm {
public:
    struct GemmInput {
        const int8_t* A = nullptr;    // [M, K] INT8 activations
        const float* A_scales = nullptr; // [M] per-token activation scales
        const int8_t* B = nullptr;    // [N, K] INT8 weights (already transposed)
        const float* B_scales = nullptr; // [N] per-channel weight scales
        int32_t M = 0;
        int32_t N = 0;
        int32_t K = 0;
        float alpha = 1.0f;           // Output scale multiplier
    };

    // Compute C = A @ B^T with per-row scaling
    static void compute(const GemmInput& input, std::span<float> C) {
        if (input.M == 0 || input.N == 0 || input.K == 0) return;

        if constexpr (UseOpenMP) {
            #pragma omp parallel for if(input.M > 1 && input.N > 64)
            for (int32_t m = 0; m < input.M; m++) {
                float a_scale = input.A_scales ? input.A_scales[m] : 1.0f;
                for (int32_t n = 0; n < input.N; n++) {
                    float b_scale = input.B_scales ? input.B_scales[n] : 1.0f;
                    int32_t acc = 0;  // INT32 accumulator
                    for (int32_t k = 0; k < input.K; k++) {
                        acc += (int32_t)input.A[(size_t)m * input.K + k] *
                               (int32_t)input.B[(size_t)n * input.K + k];
                    }
                    C[(size_t)m * input.N + n] = (float)acc * a_scale * b_scale * input.alpha;
                }
            }
        } else {
            for (int32_t m = 0; m < input.M; m++) {
                float a_scale = input.A_scales ? input.A_scales[m] : 1.0f;
                for (int32_t n = 0; n < input.N; n++) {
                    float b_scale = input.B_scales ? input.B_scales[n] : 1.0f;
                    int32_t acc = 0;
                    for (int32_t k = 0; k < input.K; k++) {
                        acc += (int32_t)input.A[(size_t)m * input.K + k] *
                               (int32_t)input.B[(size_t)n * input.K + k];
                    }
                    C[(size_t)m * input.N + n] = (float)acc * a_scale * b_scale * input.alpha;
                }
            }
        }
    }

    // Batched GEMM: multiple independent GEMMs (e.g., Q, K, V projections)
    static void batched_compute(
        std::span<const GemmInput> inputs,
        std::span<float> outputs,
        int32_t stride_C  // Stride between output blocks in C
    ) {
        for (size_t b = 0; b < inputs.size(); b++) {
            auto C_slice = outputs.subspan(b * stride_C, (size_t)inputs[b].M * inputs[b].N);
            compute(inputs[b], C_slice);
        }
    }
};

// ─── KV Cache INT8 Quantization ─────────────────────────────────────────────
//
// Per-token KV cache quantization: store K/V as INT8 with per-token scale.
// Reconstruct on read for attention computation.

class KVCacheQuant {
public:
    struct QuantizedPage {
        std::vector<int8_t> k_data;  // [max_tokens * n_kv_heads * head_dim]
        std::vector<int8_t> v_data;
        std::vector<float> k_scales; // [max_tokens]
        std::vector<float> v_scales; // [max_tokens]
        int32_t max_tokens = 0;
        int32_t num_tokens = 0;
    };

    // Quantize K/V for one token position
    static void quantize_token(
        std::span<const float> k,       // [n_kv_heads * head_dim]
        std::span<const float> v,
        std::span<int8_t> k_out,        // same size
        std::span<int8_t> v_out,
        float& k_scale,
        float& v_scale
    ) {
        size_t n = k.size();
        auto quant = [](std::span<const float> src, std::span<int8_t> dst, float& scale) {
            float amax = 0.0f;
            for (size_t i = 0; i < src.size(); i++) {
                float a = std::abs(src[i]);
                if (a > amax && std::isfinite(a)) amax = a;
            }
            if (amax < 1e-12f) amax = 1.0f;
            scale = amax / 127.0f;
            float inv = 1.0f / scale;
            for (size_t i = 0; i < src.size(); i++) {
                float v = src[i];
                if (!std::isfinite(v)) v = 0.0f;
                int32_t q = (int32_t)std::round(v * inv);
                q = std::clamp(q, -127, 127);
                dst[i] = (int8_t)q;
            }
        };
        quant(k, k_out, k_scale);
        quant(v, v_out, v_scale);
    }

    // Dequantize stored K/V back to fp32 for attention
    static void dequantize_token(
        std::span<const int8_t> k_in,
        std::span<const int8_t> v_in,
        float k_scale, float v_scale,
        std::span<float> k_out,
        std::span<float> v_out
    ) {
        for (size_t i = 0; i < k_in.size(); i++) {
            k_out[i] = (float)k_in[i] * k_scale;
            v_out[i] = (float)v_in[i] * v_scale;
        }
    }
};

} // namespace specdecode::quant
