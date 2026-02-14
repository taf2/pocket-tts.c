// tests/test_mps.c - MPS backend validation tests
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../ptts_mps.h"

#define EPSILON 1e-4f

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { \
    test_count++; \
    printf("Testing %s... ", name); \
} while(0)

#define PASS() do { \
    pass_count++; \
    printf("PASS\n"); \
} while(0)

#define FAIL(msg) do { \
    printf("FAIL: %s\n", msg); \
} while(0)

static int approx_equal(float a, float b) {
    return fabsf(a - b) < EPSILON || fabsf(a - b) / fmaxf(fabsf(a), fabsf(b)) < EPSILON;
}

static int test_linear(void) {
    TEST("linear forward");

    // Simple 2x3 @ 3x4 matmul
    float x[] = {1, 2, 3, 4, 5, 6};  // [2, 3]
    float w[] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1};  // [4, 3]
    float b[] = {0.1f, 0.2f, 0.3f, 0.4f};  // [4]
    float y[8];  // [2, 4]

    // Expected: y = x @ w^T + b
    // Row 0: [1,2,3] @ [[1,0,0,1],[0,1,0,1],[0,0,1,1]]^T + b = [1,2,3,6] + b = [1.1, 2.2, 3.3, 6.4]
    // Row 1: [4,5,6] @ ... + b = [4,5,6,15] + b = [4.1, 5.2, 6.3, 15.4]
    float expected[] = {1.1f, 2.2f, 3.3f, 6.4f, 4.1f, 5.2f, 6.3f, 15.4f};

    if (ptts_mps_linear_forward(y, x, w, b, 2, 3, 4) != 0) {
        FAIL("MPS linear forward failed");
        return 0;
    }

    for (int i = 0; i < 8; i++) {
        if (!approx_equal(y[i], expected[i])) {
            char msg[128];
            snprintf(msg, sizeof(msg), "y[%d]=%.4f, expected %.4f", i, y[i], expected[i]);
            FAIL(msg);
            return 0;
        }
    }

    PASS();
    return 1;
}

static int test_silu(void) {
    TEST("silu activation");

    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float expected[5];

    // silu(x) = x / (1 + exp(-x))
    for (int i = 0; i < 5; i++) {
        expected[i] = x[i] / (1.0f + expf(-x[i]));
    }

    if (ptts_mps_silu_forward(x, 5) != 0) {
        FAIL("MPS silu failed");
        return 0;
    }

    for (int i = 0; i < 5; i++) {
        if (!approx_equal(x[i], expected[i])) {
            char msg[128];
            snprintf(msg, sizeof(msg), "x[%d]=%.4f, expected %.4f", i, x[i], expected[i]);
            FAIL(msg);
            return 0;
        }
    }

    PASS();
    return 1;
}

static int test_rmsnorm(void) {
    TEST("rmsnorm");

    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};  // [1, 4]
    float gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4];

    // rms = sqrt(mean(x^2)) = sqrt((1+4+9+16)/4) = sqrt(7.5)
    // y = x / rms * gamma
    float rms = sqrtf(7.5f);
    float expected[] = {1.0f/rms, 2.0f/rms, 3.0f/rms, 4.0f/rms};

    if (ptts_mps_rmsnorm_forward(y, x, gamma, 1, 4, 1e-5f) != 0) {
        FAIL("MPS rmsnorm failed");
        return 0;
    }

    for (int i = 0; i < 4; i++) {
        if (!approx_equal(y[i], expected[i])) {
            char msg[128];
            snprintf(msg, sizeof(msg), "y[%d]=%.4f, expected %.4f", i, y[i], expected[i]);
            FAIL(msg);
            return 0;
        }
    }

    PASS();
    return 1;
}

int main(void) {
    printf("=== MPS Backend Validation Tests ===\n\n");

    if (ptts_mps_init() != 0) {
        printf("SKIP: MPS not available\n");
        return 0;
    }

    test_linear();
    test_silu();
    test_rmsnorm();

    printf("\n=== Results: %d/%d tests passed ===\n", pass_count, test_count);

    ptts_mps_cleanup();

    return (pass_count == test_count) ? 0 : 1;
}
