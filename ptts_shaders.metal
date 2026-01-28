#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Element-wise activations
// ============================================================================

kernel void elu_kernel(device float *x [[buffer(0)]],
                       constant float &alpha [[buffer(1)]],
                       uint idx [[thread_position_in_grid]]) {
    float val = x[idx];
    x[idx] = val >= 0.0f ? val : alpha * (exp(val) - 1.0f);
}

kernel void silu_kernel(device float *x [[buffer(0)]],
                        uint idx [[thread_position_in_grid]]) {
    float val = x[idx];
    x[idx] = val / (1.0f + exp(-val));
}

// ============================================================================
// Normalization kernels
// ============================================================================

// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Processes one row (d elements) per threadgroup
kernel void layernorm_kernel(device const float *x [[buffer(0)]],
                             device float *y [[buffer(1)]],
                             device const float *gamma [[buffer(2)]],
                             device const float *beta [[buffer(3)]],
                             constant int &d [[buffer(4)]],
                             constant float &eps [[buffer(5)]],
                             uint row [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float shared_sum[256];
    threadgroup float shared_sum_sq[256];

    const device float *row_x = x + row * d;
    device float *row_y = y + row * d;

    // Parallel reduction for mean and variance
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < uint(d); i += tg_size) {
        float val = row_x[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float mean = shared_sum[0] / float(d);
    float var = shared_sum_sq[0] / float(d) - mean * mean;
    float inv_std = rsqrt(var + eps);

    // Normalize
    for (uint i = tid; i < uint(d); i += tg_size) {
        float val = row_x[i];
        row_y[i] = gamma[i] * (val - mean) * inv_std + beta[i];
    }
}

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * gamma
kernel void rmsnorm_kernel(device const float *x [[buffer(0)]],
                           device float *y [[buffer(1)]],
                           device const float *gamma [[buffer(2)]],
                           constant int &d [[buffer(3)]],
                           constant float &eps [[buffer(4)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float shared_sum_sq[256];

    const device float *row_x = x + row * d;
    device float *row_y = y + row * d;

    // Parallel reduction for sum of squares
    float local_sum_sq = 0.0f;
    for (uint i = tid; i < uint(d); i += tg_size) {
        float val = row_x[i];
        local_sum_sq += val * val;
    }

    shared_sum_sq[tid] = local_sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_sum_sq[0] / float(d) + eps);

    // Scale
    for (uint i = tid; i < uint(d); i += tg_size) {
        row_y[i] = row_x[i] * rms * gamma[i];
    }
}

// ============================================================================
// Attention kernels
// ============================================================================

// Attention scores: scores[i,j] = sum_k(Q[i,k] * K[j,k]) / sqrt(d)
// Q: [n, d], K: [m, d], scores: [n, m]
kernel void attn_scores_kernel(device const float *Q [[buffer(0)]],
                               device const float *K [[buffer(1)]],
                               device float *scores [[buffer(2)]],
                               constant int &n [[buffer(3)]],
                               constant int &m [[buffer(4)]],
                               constant int &d [[buffer(5)]],
                               constant float &scale [[buffer(6)]],
                               constant int &causal [[buffer(7)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.y;  // query index
    uint j = gid.x;  // key index

    if (i >= uint(n) || j >= uint(m)) return;

    // Causal masking: mask out future positions
    if (causal && j > i) {
        scores[i * m + j] = -INFINITY;
        return;
    }

    float dot = 0.0f;
    for (int k = 0; k < d; k++) {
        dot += Q[i * d + k] * K[j * d + k];
    }
    scores[i * m + j] = dot * scale;
}

// Row-wise softmax
kernel void attn_softmax_kernel(device float *scores [[buffer(0)]],
                                constant int &n [[buffer(1)]],
                                constant int &m [[buffer(2)]],
                                uint row [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tg_size [[threads_per_threadgroup]]) {
    threadgroup float shared_max[256];
    threadgroup float shared_sum[256];

    device float *row_scores = scores + row * m;

    // Find max for numerical stability
    float local_max = -INFINITY;
    for (uint j = tid; j < uint(m); j += tg_size) {
        local_max = max(local_max, row_scores[j]);
    }
    shared_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];

    // Compute exp and sum
    float local_sum = 0.0f;
    for (uint j = tid; j < uint(m); j += tg_size) {
        float val = exp(row_scores[j] - row_max);
        row_scores[j] = val;
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = shared_sum[0];

    // Normalize
    float inv_sum = 1.0f / row_sum;
    for (uint j = tid; j < uint(m); j += tg_size) {
        row_scores[j] *= inv_sum;
    }
}

// Apply attention: out[i,k] = sum_j(scores[i,j] * V[j,k])
kernel void attn_apply_kernel(device const float *scores [[buffer(0)]],
                              device const float *V [[buffer(1)]],
                              device float *out [[buffer(2)]],
                              constant int &n [[buffer(3)]],
                              constant int &m [[buffer(4)]],
                              constant int &d [[buffer(5)]],
                              uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.y;  // query index
    uint k = gid.x;  // feature index

    if (i >= uint(n) || k >= uint(d)) return;

    float sum = 0.0f;
    for (int j = 0; j < m; j++) {
        sum += scores[i * m + j] * V[j * d + k];
    }
    out[i * d + k] = sum;
}

// ============================================================================
// Convolution kernels
// ============================================================================

// Conv1d: y[c_out, t_out] = sum over c_in, k of x[c_in, t*stride+k] * w[c_out, c_in, k]
kernel void conv1d_kernel(device const float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device const float *b [[buffer(2)]],
                          device float *y [[buffer(3)]],
                          constant int &in_ch [[buffer(4)]],
                          constant int &out_ch [[buffer(5)]],
                          constant int &T [[buffer(6)]],
                          constant int &k [[buffer(7)]],
                          constant int &stride [[buffer(8)]],
                          constant int &groups [[buffer(9)]],
                          constant int &out_len [[buffer(10)]],
                          constant int &has_bias [[buffer(11)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint oc = gid.y;  // output channel
    uint t = gid.x;   // output time position

    if (oc >= uint(out_ch) || t >= uint(out_len)) return;

    int ch_per_group_in = in_ch / groups;
    int ch_per_group_out = out_ch / groups;
    int g = oc / ch_per_group_out;

    float sum = 0.0f;
    for (int ic = 0; ic < ch_per_group_in; ic++) {
        int ic_global = g * ch_per_group_in + ic;
        for (int ki = 0; ki < k; ki++) {
            int t_in = int(t) * stride + ki;
            if (t_in >= 0 && t_in < T) {
                // x: [in_ch, T], w: [out_ch, ch_per_group_in, k]
                float x_val = x[ic_global * T + t_in];
                float w_val = w[oc * ch_per_group_in * k + ic * k + ki];
                sum += x_val * w_val;
            }
        }
    }

    if (has_bias) {
        sum += b[oc];
    }
    y[oc * out_len + t] = sum;
}

// ConvTranspose1d (deconvolution)
kernel void convtr1d_kernel(device const float *x [[buffer(0)]],
                            device const float *w [[buffer(1)]],
                            device const float *b [[buffer(2)]],
                            device float *y [[buffer(3)]],
                            constant int &in_ch [[buffer(4)]],
                            constant int &out_ch [[buffer(5)]],
                            constant int &T [[buffer(6)]],
                            constant int &k [[buffer(7)]],
                            constant int &stride [[buffer(8)]],
                            constant int &groups [[buffer(9)]],
                            constant int &out_len [[buffer(10)]],
                            constant int &has_bias [[buffer(11)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint oc = gid.y;
    uint t_out = gid.x;

    if (oc >= uint(out_ch) || t_out >= uint(out_len)) return;

    int ch_per_group_in = in_ch / groups;
    int ch_per_group_out = out_ch / groups;
    int g = oc / ch_per_group_out;
    int oc_in_group = oc % ch_per_group_out;

    float sum = 0.0f;
    for (int ic = 0; ic < ch_per_group_in; ic++) {
        int ic_global = g * ch_per_group_in + ic;
        for (int ki = 0; ki < k; ki++) {
            // For transposed conv: t_out = t_in * stride + ki
            // So: t_in = (t_out - ki) / stride, must be integer and in range
            int t_in_shifted = int(t_out) - ki;
            if (t_in_shifted >= 0 && t_in_shifted % stride == 0) {
                int t_in = t_in_shifted / stride;
                if (t_in < T) {
                    // w: [in_ch, ch_per_group_out, k]
                    float x_val = x[ic_global * T + t_in];
                    float w_val = w[ic_global * ch_per_group_out * k + oc_in_group * k + ki];
                    sum += x_val * w_val;
                }
            }
        }
    }

    if (has_bias) {
        sum += b[oc];
    }
    y[oc * out_len + t_out] = sum;
}
