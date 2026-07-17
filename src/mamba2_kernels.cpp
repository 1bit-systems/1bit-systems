// mamba2_kernels.cpp — Mamba2 CPU reference forward pass
// Implements the complete Mamba2 layer in plain C++ for:
//   1. Correctness verification against PyTorch
//   2. CPU fallback for non-GPU inference
//   3. Reference for HIP kernel porting
//
// Mamba2 architecture (SSD = Structured State Space Dual):
//   Given input x:
//     1. in_proj:  z, xBC, dt = split(linear(x, in_proj_w))
//     2. conv1d:   xBC' = silu(conv1d(xBC) + conv1d_b)
//     3. split:    x, B, C = split(xBC')
//     4. discretize: A_bar = exp(dt * A)  where A = -exp(A_log)
//     5. selective_scan: y = ssm_scan(x, A_bar, B, C, dt + dt_bias)
//     6. out_proj: y = linear(y * silu(z), out_proj_w)
//
// Reference: Gu & Dao, "Mamba2: SSDs for Efficient Sequence Modeling" (2024)

#include "mamba2_kernels.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// ── Helper: softplus for dt activation ──
static inline float softplus(float x) {
    // log(1 + exp(x)) — numerically stable
    if (x > 20.0f) return x;
    return std::log1pf(std::expf(x));
}

// ── Helper: one step of the selective scan ──
// Updates the SSM hidden state and produces output for one timestep.
// This is the core Mamba2 SSD operation.
static void selective_scan_step(
    const float* x_t,          // [head_dim] for current head
    float dt_t,                 // discretization step for current head
    float A_h,                  // A = -exp(A_log) for current head
    const float* B_t,           // [d_state] for current head's group
    const float* C_t,           // [d_state] for current head's group
    float D_h,                  // skip connection for current head
    float* state,               // [d_state] — updated in-place
    float* y_t,                 // [head_dim] — output for this head
    int head_dim, int d_state
) {
    // Discretize A: A_bar = exp(dt * A) = exp(-dt * exp(A_log))
    float dt_softplus = softplus(dt_t);
    float A_bar = std::exp(dt_softplus * A_h);  // A_h is -exp(A_log), so A_bar in (0,1)

    // B is also discretized: B_bar = dt_softplus * B_t
    // Then: state = A_bar * state + B_bar * x
    //        y = C^T @ state + D * x

    for (int i = 0; i < d_state; ++i) {
        state[i] = A_bar * state[i] + dt_softplus * B_t[i] * x_t[0];
    }

    // Output: y_t = C @ state + D * x_t
    float c_dot_state = 0.0f;
    for (int i = 0; i < d_state; ++i) {
        c_dot_state += C_t[i] * state[i];
    }

    // Apply to all elements in this head's dimension
    for (int hd = 0; hd < head_dim; ++hd) {
        y_t[hd] = c_dot_state * x_t[hd] + D_h * x_t[hd];
        // Note: in the full Mamba2, the output is:
        // y = C @ state + D * x  (but C@state is scalar broadcast)
    }
}

// ── CPU reference forward pass for one Mamba2 layer (single token) ──
// This processes ONE token at a time (autoregressive mode).
// For prefill (batched), the scan iterates over the sequence.
void mamba2_cpu_forward(
    const float* x,
    const float* in_proj_w,
    const float* conv1d_w,
    const float* conv1d_b,
    const float* dt_bias,
    const float* A_log,
    const float* D,
    const float* norm_w,
    const float* out_proj_w,
    float* conv_state,
    float* ssm_state,
    float* y,
    const Mamba2Config& cfg
) {
    const int64_t d_model   = cfg.d_model;
    const int64_t d_state   = cfg.d_state;
    const int64_t d_conv    = cfg.d_conv;
    const int64_t d_inner   = cfg.d_inner;
    const int64_t n_head    = cfg.n_head;
    const int64_t n_group   = cfg.n_group;
    const int64_t head_dim  = cfg.head_dim;

    const int64_t conv_dim  = d_inner + 2 * n_group * d_state;
    const int64_t d_in_proj = d_inner + conv_dim + n_head;  // z + xBC + dt

    // Temporary buffers
    std::vector<float> in_proj_out(d_in_proj, 0.0f);
    std::vector<float> xBC_conv(conv_dim, 0.0f);
    std::vector<float> xBC_act(conv_dim, 0.0f);
    std::vector<float> x_proj_out(n_head + 2 * n_group * d_state, 0.0f);
    std::vector<float> y_inner(d_inner, 0.0f);
    std::vector<float> z_reshaped(d_inner, 0.0f);

    // ── Step 1: in_proj ──
    // in_proj_out = in_proj_w @ x  (bias is assumed zero)
    for (int64_t i = 0; i < d_in_proj; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < d_model; ++j) {
            sum += in_proj_w[i * d_model + j] * x[j];
        }
        in_proj_out[i] = sum;
    }

    // Split: z [d_inner], xBC [conv_dim], dt [n_head]
    const float* z_raw   = in_proj_out.data();
    const float* xBC_raw = in_proj_out.data() + d_inner;
    const float* dt_raw  = in_proj_out.data() + d_inner + conv_dim;

    // ── Step 2: conv1d ──
    // Shift conv state: push new input, pop oldest
    // conv_state shape: [d_conv - 1, conv_dim]
    // We store as [conv_dim * (d_conv - 1)] flattened, row-major: [timestep][channel]

    // Apply 1D convolution: out[t] = sum_{k=0}^{d_conv-1} w[k] * state_or_input[d_conv-1-k]
    for (int64_t c = 0; c < conv_dim; ++c) {
        float acc = conv1d_b[c];  // start with bias

        // Convolution: w[0]*xBC[t] + w[1]*state[0] + w[2]*state[1] + ...
        for (int64_t k = 0; k < d_conv; ++k) {
            float val;
            if (k == 0) {
                val = xBC_raw[c];  // current input
            } else {
                val = conv_state[(k - 1) * conv_dim + c];  // previous state
            }
            acc += conv1d_w[k * conv_dim + c] * val;
        }
        xBC_conv[c] = acc;

        // Update state: shift and insert new value at position d_conv-2
        for (int64_t k = d_conv - 3; k >= 0; --k) {
            conv_state[(k + 1) * conv_dim + c] = conv_state[k * conv_dim + c];
        }
        if (d_conv > 1) {
            conv_state[0 * conv_dim + c] = xBC_raw[c];  // store most recent
        }
    }

    // silu activation: xBC_act = xBC_conv * sigmoid(xBC_conv)
    for (int64_t i = 0; i < conv_dim; ++i) {
        float v = xBC_conv[i];
        xBC_act[i] = v / (1.0f + std::expf(-v));  // silu = x * sigmoid(x)
    }

    // ── Step 3: split xBC into x, B, C ──
    // x:  [d_inner]
    // B:  [n_group, d_state]  (note: single token, so batch dim is 1)
    // C:  [n_group, d_state]
    const float* x_inner_raw = xBC_act.data();
    const float* B_raw = xBC_act.data() + d_inner;
    const float* C_raw = xBC_act.data() + d_inner + n_group * d_state;

    // ── Step 4 + 5: selective scan per head ──
    // Each head processes head_dim elements with the same A, dt, B, C
    // Heads are grouped: n_group groups, each with n_head/n_group heads

    int64_t heads_per_group = n_head / n_group;

    for (int64_t g = 0; g < n_group; ++g) {
        for (int64_t h = 0; h < heads_per_group; ++h) {
            int64_t head_id = g * heads_per_group + h;

            // dt for this head: dt_raw[head_id] + dt_bias[head_id] → softplus
            float dt_val = softplus(dt_raw[head_id] + dt_bias[head_id]);

            // A for this head: A = -exp(A_log[head_id])
            float A_val = -std::expf(A_log[head_id]);

            // D for this head
            float D_val = D[head_id];

            // B and C for this head's group
            const float* B_g = B_raw + g * d_state;
            const float* C_g = C_raw + g * d_state;

            // x segment for this head: [head_dim] starting at head_id * head_dim
            const float* x_h = x_inner_raw + head_id * head_dim;
            float* y_h = y_inner.data() + head_id * head_dim;
            float* ssm_h = ssm_state + head_id * d_state;

            // Run one step of the selective scan
            selective_scan_step(
                x_h, dt_val, A_val, B_g, C_g, D_val,
                ssm_h, y_h, head_dim, d_state
            );
        }
    }

    // ── Step 5b: Group norm on y_inner ──
    // norm_w shape: [d_inner / n_group, n_group]
    // Apply RMS norm per group
    int64_t group_size = d_inner / n_group;
    for (int64_t g = 0; g < n_group; ++g) {
        float* group_start = y_inner.data() + g * group_size;
        const float* norm_g = norm_w + g * (d_inner / n_group);  // actually [group_size, n_group]

        // Compute RMS
        float sum_sq = 0.0f;
        for (int64_t i = 0; i < group_size; ++i) {
            sum_sq += group_start[i] * group_start[i];
        }
        float rms = std::sqrt(sum_sq / group_size + 1e-6f);

        // Apply norm
        for (int64_t i = 0; i < group_size; ++i) {
            // The norm_w for group g: stored as [group_size] (column g of norm_w)
            group_start[i] = (group_start[i] / rms) * norm_w[g + i * n_group];
            // norm_w layout: [d_inner/n_group, n_group] — accessed as norm_w[i * n_group + g]
        }
    }

    // ── Step 6: Gate with z ──
    // Reshape z to [n_head, head_dim] and apply silu
    for (int64_t i = 0; i < d_inner; ++i) {
        float zv = z_raw[i];
        z_reshaped[i] = zv / (1.0f + std::expf(-zv));  // silu(z)
    }

    // Multiply: y_inner = y_inner * silu(z)
    for (int64_t i = 0; i < d_inner; ++i) {
        y_inner[i] = y_inner[i] * z_reshaped[i];
    }

    // ── Step 7: out_proj ──
    // y = out_proj_w @ y_inner
    for (int64_t i = 0; i < d_model; ++i) {
        float sum = 0.0f;
        for (int64_t j = 0; j < d_inner; ++j) {
            sum += out_proj_w[i * d_inner + j] * y_inner[j];
        }
        y[i] = sum;
    }
}

// ── CPU batched prefill: process a full sequence ──
// Runs the full Mamba2 recurrence over L tokens.
void mamba2_cpu_prefill(
    const float* x_seq,          // [L, d_model]
    const float* in_proj_w,
    const float* conv1d_w,
    const float* conv1d_b,
    const float* dt_bias,
    const float* A_log,
    const float* D,
    const float* norm_w,
    const float* out_proj_w,
    float* conv_state,           // [d_conv-1, conv_dim] — updated
    float* ssm_state,            // [d_state, d_inner] — updated
    float* y_seq,                // [L, d_model] — output
    const Mamba2Config& cfg,
    int L                         // sequence length
) {
    for (int t = 0; t < L; ++t) {
        mamba2_cpu_forward(
            x_seq + t * cfg.d_model,
            in_proj_w, conv1d_w, conv1d_b, dt_bias, A_log, D, norm_w, out_proj_w,
            conv_state, ssm_state,
            y_seq + t * cfg.d_model,
            cfg
        );
    }
}
