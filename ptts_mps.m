// ptts_mps.m - Metal Performance Shaders backend for pocket-tts
#ifdef PTTS_USE_MPS

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ptts_mps.h"

// ============================================================================
// Global state
// ============================================================================

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;

// Compute pipeline states
static id<MTLComputePipelineState> g_elu_pipeline = nil;
static id<MTLComputePipelineState> g_silu_pipeline = nil;
static id<MTLComputePipelineState> g_layernorm_pipeline = nil;
static id<MTLComputePipelineState> g_rmsnorm_pipeline = nil;
static id<MTLComputePipelineState> g_attn_scores_pipeline = nil;
static id<MTLComputePipelineState> g_attn_softmax_pipeline = nil;
static id<MTLComputePipelineState> g_attn_apply_pipeline = nil;
static id<MTLComputePipelineState> g_conv1d_pipeline = nil;
static id<MTLComputePipelineState> g_convtr1d_pipeline = nil;

// Batch mode state
static int g_batch_mode = 0;
static id<MTLCommandBuffer> g_batch_cmd = nil;

// ============================================================================
// Weight cache - maps CPU pointer to GPU buffer
// ============================================================================

#define WEIGHT_CACHE_SIZE 512

typedef struct {
    const void *cpu_ptr;
    id<MTLBuffer> gpu_buffer;
    size_t size;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[WEIGHT_CACHE_SIZE];
static int g_weight_cache_count = 0;

static id<MTLBuffer> get_cached_buffer(const void *cpu_ptr, size_t size) {
    // Look up in cache
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].cpu_ptr == cpu_ptr && g_weight_cache[i].size == size) {
            return g_weight_cache[i].gpu_buffer;
        }
    }

    // Not found - create new buffer
    id<MTLBuffer> buffer = [g_device newBufferWithBytes:cpu_ptr
                                                 length:size
                                                options:MTLResourceStorageModeShared];

    if (buffer && g_weight_cache_count < WEIGHT_CACHE_SIZE) {
        g_weight_cache[g_weight_cache_count].cpu_ptr = cpu_ptr;
        g_weight_cache[g_weight_cache_count].gpu_buffer = buffer;
        g_weight_cache[g_weight_cache_count].size = size;
        g_weight_cache_count++;
    }

    return buffer;
}

// ============================================================================
// Buffer pool for temporary allocations
// ============================================================================

#define BUFFER_POOL_SIZE 64

typedef struct {
    id<MTLBuffer> buffer;
    size_t size;
    int in_use;
} pool_buffer_t;

static pool_buffer_t g_buffer_pool[BUFFER_POOL_SIZE];

static id<MTLBuffer> pool_get_buffer(size_t size) {
    // Round up to 64KB alignment for efficiency
    size_t aligned_size = (size + 65535) & ~65535;

    // Look for existing buffer of sufficient size
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (!g_buffer_pool[i].in_use && g_buffer_pool[i].size >= aligned_size) {
            g_buffer_pool[i].in_use = 1;
            return g_buffer_pool[i].buffer;
        }
    }

    // Allocate new buffer
    id<MTLBuffer> buffer = [g_device newBufferWithLength:aligned_size
                                                 options:MTLResourceStorageModeShared];

    // Add to pool
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (g_buffer_pool[i].buffer == nil) {
            g_buffer_pool[i].buffer = buffer;
            g_buffer_pool[i].size = aligned_size;
            g_buffer_pool[i].in_use = 1;
            break;
        }
    }

    return buffer;
}

static void pool_release_buffer(id<MTLBuffer> buffer) {
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (g_buffer_pool[i].buffer == buffer) {
            g_buffer_pool[i].in_use = 0;
            return;
        }
    }
}

// ============================================================================
// Initialization
// ============================================================================

static int load_shader_library(void) {
    NSError *error = nil;

    // Try to load from compiled metallib first
    NSString *libPath = [[NSBundle mainBundle] pathForResource:@"ptts_shaders" ofType:@"metallib"];
    if (libPath) {
        g_library = [g_device newLibraryWithFile:libPath error:&error];
        if (g_library) return 0;
    }

    // Fall back to compiling from source
    NSString *shaderPath = @"ptts_shaders.metal";
    NSString *shaderSource = [NSString stringWithContentsOfFile:shaderPath
                                                       encoding:NSUTF8StringEncoding
                                                          error:&error];
    if (!shaderSource) {
        fprintf(stderr, "MPS: Failed to load shader source: %s\n",
                [[error localizedDescription] UTF8String]);
        return -1;
    }

    MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;

    g_library = [g_device newLibraryWithSource:shaderSource options:options error:&error];
    if (!g_library) {
        fprintf(stderr, "MPS: Failed to compile shaders: %s\n",
                [[error localizedDescription] UTF8String]);
        return -1;
    }

    return 0;
}

static id<MTLComputePipelineState> create_pipeline(NSString *functionName) {
    NSError *error = nil;
    id<MTLFunction> function = [g_library newFunctionWithName:functionName];
    if (!function) {
        fprintf(stderr, "MPS: Function '%s' not found\n", [functionName UTF8String]);
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [g_device newComputePipelineStateWithFunction:function
                                                                                   error:&error];
    if (!pipeline) {
        fprintf(stderr, "MPS: Failed to create pipeline for '%s': %s\n",
                [functionName UTF8String], [[error localizedDescription] UTF8String]);
        return nil;
    }

    return pipeline;
}

int ptts_mps_init(void) {
    @autoreleasepool {
        // Get default Metal device
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            fprintf(stderr, "MPS: No Metal device found\n");
            return -1;
        }

        // Check for Apple Silicon (MPS works best on Apple Silicon)
        if (![g_device supportsFamily:MTLGPUFamilyApple7]) {
            fprintf(stderr, "MPS: Warning - device may not support all MPS features\n");
        }

        // Create command queue
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "MPS: Failed to create command queue\n");
            return -1;
        }

        // Load shader library
        if (load_shader_library() != 0) {
            return -1;
        }

        // Create compute pipelines
        g_elu_pipeline = create_pipeline(@"elu_kernel");
        g_silu_pipeline = create_pipeline(@"silu_kernel");
        g_layernorm_pipeline = create_pipeline(@"layernorm_kernel");
        g_rmsnorm_pipeline = create_pipeline(@"rmsnorm_kernel");
        g_attn_scores_pipeline = create_pipeline(@"attn_scores_kernel");
        g_attn_softmax_pipeline = create_pipeline(@"attn_softmax_kernel");
        g_attn_apply_pipeline = create_pipeline(@"attn_apply_kernel");
        g_conv1d_pipeline = create_pipeline(@"conv1d_kernel");
        g_convtr1d_pipeline = create_pipeline(@"convtr1d_kernel");

        // Initialize caches
        memset(g_weight_cache, 0, sizeof(g_weight_cache));
        memset(g_buffer_pool, 0, sizeof(g_buffer_pool));

        fprintf(stderr, "MPS: Initialized on %s\n", [[g_device name] UTF8String]);
        return 0;
    }
}

void ptts_mps_cleanup(void) {
    @autoreleasepool {
        // Clear weight cache
        for (int i = 0; i < g_weight_cache_count; i++) {
            g_weight_cache[i].gpu_buffer = nil;
        }
        g_weight_cache_count = 0;

        // Clear buffer pool
        for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
            g_buffer_pool[i].buffer = nil;
            g_buffer_pool[i].in_use = 0;
        }

        // Release pipelines
        g_elu_pipeline = nil;
        g_silu_pipeline = nil;
        g_layernorm_pipeline = nil;
        g_rmsnorm_pipeline = nil;
        g_attn_scores_pipeline = nil;
        g_attn_softmax_pipeline = nil;
        g_attn_apply_pipeline = nil;
        g_conv1d_pipeline = nil;
        g_convtr1d_pipeline = nil;

        g_library = nil;
        g_queue = nil;
        g_device = nil;
    }
}

int ptts_mps_available(void) {
    return g_device != nil && g_queue != nil;
}

void ptts_mps_clear_weight_cache(void) {
    for (int i = 0; i < g_weight_cache_count; i++) {
        g_weight_cache[i].gpu_buffer = nil;
    }
    g_weight_cache_count = 0;
}

size_t ptts_mps_memory_used(void) {
    size_t total = 0;
    for (int i = 0; i < g_weight_cache_count; i++) {
        total += g_weight_cache[i].size;
    }
    for (int i = 0; i < BUFFER_POOL_SIZE; i++) {
        if (g_buffer_pool[i].buffer) {
            total += g_buffer_pool[i].size;
        }
    }
    return total;
}

// ============================================================================
// Batch execution
// ============================================================================

void ptts_mps_begin_batch(void) {
    g_batch_mode = 1;
    g_batch_cmd = [g_queue commandBuffer];
}

void ptts_mps_end_batch(void) {
    if (g_batch_cmd) {
        [g_batch_cmd commit];
        [g_batch_cmd waitUntilCompleted];
        g_batch_cmd = nil;
    }
    g_batch_mode = 0;
}

static id<MTLCommandBuffer> get_command_buffer(void) {
    if (g_batch_mode && g_batch_cmd) {
        return g_batch_cmd;
    }
    return [g_queue commandBuffer];
}

static void submit_if_not_batch(id<MTLCommandBuffer> cmd) {
    if (!g_batch_mode) {
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

// ============================================================================
// Linear layer: y = x @ W^T + b
// ============================================================================

int ptts_mps_linear_forward(float *y, const float *x, const float *w,
                            const float *b, int n, int in_features, int out_features) {
    @autoreleasepool {
        if (!ptts_mps_available()) return -1;

        // For small matrices, use Accelerate (CPU BLAS) - less overhead
        if (n * in_features < 4096) {
            // y = x @ W^T using cblas_sgemm
            // C = alpha * A * B^T + beta * C
            // A: [n, in], B: [out, in], C: [n, out]
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        n, out_features, in_features,
                        1.0f, x, in_features,
                        w, in_features,
                        0.0f, y, out_features);

            // Add bias
            if (b) {
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < out_features; j++) {
                        y[i * out_features + j] += b[j];
                    }
                }
            }
            return 0;
        }

        // For larger matrices, use MPS
        size_t x_size = n * in_features * sizeof(float);
        size_t w_size = out_features * in_features * sizeof(float);
        size_t y_size = n * out_features * sizeof(float);

        // Get or create GPU buffers
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:x_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = get_cached_buffer(w, w_size);
        id<MTLBuffer> y_buf = pool_get_buffer(y_size);

        if (!x_buf || !w_buf || !y_buf) {
            return -1;
        }

        // Create matrix descriptors
        // x: [n, in_features] - row major
        MPSMatrixDescriptor *x_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:n
                             columns:in_features
                            rowBytes:in_features * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        // w: [out_features, in_features] - will be transposed
        MPSMatrixDescriptor *w_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:out_features
                             columns:in_features
                            rowBytes:in_features * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        // y: [n, out_features]
        MPSMatrixDescriptor *y_desc = [MPSMatrixDescriptor
            matrixDescriptorWithRows:n
                             columns:out_features
                            rowBytes:out_features * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *x_mat = [[MPSMatrix alloc] initWithBuffer:x_buf descriptor:x_desc];
        MPSMatrix *w_mat = [[MPSMatrix alloc] initWithBuffer:w_buf descriptor:w_desc];
        MPSMatrix *y_mat = [[MPSMatrix alloc] initWithBuffer:y_buf descriptor:y_desc];

        // Create matrix multiplication: y = x @ w^T
        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_device
               transposeLeft:NO
              transposeRight:YES
                  resultRows:n
               resultColumns:out_features
             interiorColumns:in_features
                       alpha:1.0
                        beta:0.0];

        // Execute
        id<MTLCommandBuffer> cmd = get_command_buffer();
        [matmul encodeToCommandBuffer:cmd leftMatrix:x_mat rightMatrix:w_mat resultMatrix:y_mat];
        submit_if_not_batch(cmd);

        // Copy result back
        memcpy(y, [y_buf contents], y_size);

        // Add bias (on CPU for now, could be GPU kernel)
        if (b) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < out_features; j++) {
                    y[i * out_features + j] += b[j];
                }
            }
        }

        pool_release_buffer(y_buf);
        return 0;
    }
}

// ============================================================================
// Activation functions
// ============================================================================

int ptts_mps_elu_forward(float *x, int n, float alpha) {
    @autoreleasepool {
        if (!ptts_mps_available() || !g_elu_pipeline) return -1;

        size_t size = n * sizeof(float);
        id<MTLBuffer> buf = [g_device newBufferWithBytes:x length:size
                                                 options:MTLResourceStorageModeShared];
        if (!buf) return -1;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_elu_pipeline];
        [encoder setBuffer:buf offset:0 atIndex:0];
        [encoder setBytes:&alpha length:sizeof(float) atIndex:1];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        NSUInteger threadGroupSize = MIN(256, g_elu_pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        submit_if_not_batch(cmd);
        memcpy(x, [buf contents], size);

        return 0;
    }
}

int ptts_mps_silu_forward(float *x, int n) {
    @autoreleasepool {
        if (!ptts_mps_available() || !g_silu_pipeline) return -1;

        size_t size = n * sizeof(float);
        id<MTLBuffer> buf = [g_device newBufferWithBytes:x length:size
                                                 options:MTLResourceStorageModeShared];
        if (!buf) return -1;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_silu_pipeline];
        [encoder setBuffer:buf offset:0 atIndex:0];

        MTLSize gridSize = MTLSizeMake(n, 1, 1);
        NSUInteger threadGroupSize = MIN(256, g_silu_pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        submit_if_not_batch(cmd);
        memcpy(x, [buf contents], size);

        return 0;
    }
}

// ============================================================================
// Normalization layers
// ============================================================================

int ptts_mps_layernorm_forward(float *y, const float *x, const float *gamma,
                               const float *beta, int n, int d, float eps) {
    @autoreleasepool {
        if (!ptts_mps_available() || !g_layernorm_pipeline) return -1;

        size_t x_size = n * d * sizeof(float);
        size_t param_size = d * sizeof(float);

        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:x_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> y_buf = pool_get_buffer(x_size);
        id<MTLBuffer> gamma_buf = get_cached_buffer(gamma, param_size);
        id<MTLBuffer> beta_buf = get_cached_buffer(beta, param_size);

        if (!x_buf || !y_buf || !gamma_buf || !beta_buf) return -1;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_layernorm_pipeline];
        [encoder setBuffer:x_buf offset:0 atIndex:0];
        [encoder setBuffer:y_buf offset:0 atIndex:1];
        [encoder setBuffer:gamma_buf offset:0 atIndex:2];
        [encoder setBuffer:beta_buf offset:0 atIndex:3];
        [encoder setBytes:&d length:sizeof(int) atIndex:4];
        [encoder setBytes:&eps length:sizeof(float) atIndex:5];

        // One threadgroup per row
        MTLSize gridSize = MTLSizeMake(1, n, 1);
        NSUInteger tgSize = MIN(256, g_layernorm_pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);

        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        submit_if_not_batch(cmd);
        memcpy(y, [y_buf contents], x_size);
        pool_release_buffer(y_buf);

        return 0;
    }
}

int ptts_mps_rmsnorm_forward(float *y, const float *x, const float *gamma,
                             int n, int d, float eps) {
    @autoreleasepool {
        if (!ptts_mps_available() || !g_rmsnorm_pipeline) return -1;

        size_t x_size = n * d * sizeof(float);
        size_t param_size = d * sizeof(float);

        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:x_size
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> y_buf = pool_get_buffer(x_size);
        id<MTLBuffer> gamma_buf = get_cached_buffer(gamma, param_size);

        if (!x_buf || !y_buf || !gamma_buf) return -1;

        id<MTLCommandBuffer> cmd = get_command_buffer();
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_rmsnorm_pipeline];
        [encoder setBuffer:x_buf offset:0 atIndex:0];
        [encoder setBuffer:y_buf offset:0 atIndex:1];
        [encoder setBuffer:gamma_buf offset:0 atIndex:2];
        [encoder setBytes:&d length:sizeof(int) atIndex:3];
        [encoder setBytes:&eps length:sizeof(float) atIndex:4];

        MTLSize gridSize = MTLSizeMake(1, n, 1);
        NSUInteger tgSize = MIN(256, g_rmsnorm_pipeline.maxTotalThreadsPerThreadgroup);
        MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);

        [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        submit_if_not_batch(cmd);
        memcpy(y, [y_buf contents], x_size);
        pool_release_buffer(y_buf);

        return 0;
    }
}

// ============================================================================
// Attention: out = softmax(Q @ K^T / sqrt(d)) @ V
// ============================================================================

int ptts_mps_attention_forward(float *out, const float *Q, const float *K, const float *V,
                               int n, int m, int heads, int d, int causal) {
    @autoreleasepool {
        if (!ptts_mps_available()) return -1;
        if (!g_attn_scores_pipeline || !g_attn_softmax_pipeline || !g_attn_apply_pipeline) return -1;

        // Process each head separately
        size_t qkv_head_size = n * d * sizeof(float);  // Q: [n, d] per head
        size_t kv_head_size = m * d * sizeof(float);   // K,V: [m, d] per head
        size_t scores_size = n * m * sizeof(float);    // scores: [n, m] per head

        float scale = 1.0f / sqrtf((float)d);

        for (int h = 0; h < heads; h++) {
            const float *Q_h = Q + h * n * d;
            const float *K_h = K + h * m * d;
            const float *V_h = V + h * m * d;
            float *out_h = out + h * n * d;

            // Create buffers
            id<MTLBuffer> Q_buf = [g_device newBufferWithBytes:Q_h length:qkv_head_size
                                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> K_buf = [g_device newBufferWithBytes:K_h length:kv_head_size
                                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> V_buf = [g_device newBufferWithBytes:V_h length:kv_head_size
                                                       options:MTLResourceStorageModeShared];
            id<MTLBuffer> scores_buf = pool_get_buffer(scores_size);
            id<MTLBuffer> out_buf = pool_get_buffer(qkv_head_size);

            if (!Q_buf || !K_buf || !V_buf || !scores_buf || !out_buf) {
                return -1;
            }

            id<MTLCommandBuffer> cmd = get_command_buffer();

            // Phase 1: Compute attention scores
            {
                id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
                [encoder setComputePipelineState:g_attn_scores_pipeline];
                [encoder setBuffer:Q_buf offset:0 atIndex:0];
                [encoder setBuffer:K_buf offset:0 atIndex:1];
                [encoder setBuffer:scores_buf offset:0 atIndex:2];
                [encoder setBytes:&n length:sizeof(int) atIndex:3];
                [encoder setBytes:&m length:sizeof(int) atIndex:4];
                [encoder setBytes:&d length:sizeof(int) atIndex:5];
                [encoder setBytes:&scale length:sizeof(float) atIndex:6];
                [encoder setBytes:&causal length:sizeof(int) atIndex:7];

                MTLSize gridSize = MTLSizeMake(m, n, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Phase 2: Softmax (row-wise)
            {
                id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
                [encoder setComputePipelineState:g_attn_softmax_pipeline];
                [encoder setBuffer:scores_buf offset:0 atIndex:0];
                [encoder setBytes:&n length:sizeof(int) atIndex:1];
                [encoder setBytes:&m length:sizeof(int) atIndex:2];

                MTLSize gridSize = MTLSizeMake(1, n, 1);
                NSUInteger tgSize = MIN(256, g_attn_softmax_pipeline.maxTotalThreadsPerThreadgroup);
                MTLSize threadgroupSize = MTLSizeMake(tgSize, 1, 1);
                [encoder dispatchThreadgroups:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Phase 3: Apply attention (scores @ V)
            {
                id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];
                [encoder setComputePipelineState:g_attn_apply_pipeline];
                [encoder setBuffer:scores_buf offset:0 atIndex:0];
                [encoder setBuffer:V_buf offset:0 atIndex:1];
                [encoder setBuffer:out_buf offset:0 atIndex:2];
                [encoder setBytes:&n length:sizeof(int) atIndex:3];
                [encoder setBytes:&m length:sizeof(int) atIndex:4];
                [encoder setBytes:&d length:sizeof(int) atIndex:5];

                MTLSize gridSize = MTLSizeMake(d, n, 1);
                MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            submit_if_not_batch(cmd);

            // Copy result back
            memcpy(out_h, [out_buf contents], qkv_head_size);

            pool_release_buffer(scores_buf);
            pool_release_buffer(out_buf);
        }

        return 0;
    }
}

#endif // PTTS_USE_MPS
