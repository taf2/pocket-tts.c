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

#endif // PTTS_USE_MPS
