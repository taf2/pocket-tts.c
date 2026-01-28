# Pocket-TTS Pure C (WIP)
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm
BLAS_LIBS ?= -lopenblas
CUDA_LIBS ?= -lcudart -lcublas -lnvrtc -lcuda

SRCS = ptts.c ptts_audio.c ptts_safetensors.c ptts_spm.c ptts_kernels.c ptts_flowlm.c ptts_mimi.c
OBJS = $(SRCS:.c=.o)
CUDA_OBJS = $(OBJS) ptts_cuda.o
MAIN = main.c
TARGET = ptts
LIB = libptts.a

.PHONY: all clean help cpu lib info test blas cuda cuda-validate cuda-validate-test

all: help

help:
	@echo "Pocket-TTS Pure C (WIP) - Build Targets"
	@echo ""
	@echo "  make cpu      - Pure C, no dependencies"
	@echo "  make blas     - OpenBLAS accelerated"
	@echo "  make cuda     - NVIDIA CUDA + cuBLAS accelerated"
	@echo "  make cuda-validate - CUDA build with layer-by-layer validator"
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo "  make test     - Run hello world regression test"
	@echo ""
	@echo "Example: make cpu && ./ptts --dummy -p \"hello\" -o out.wav"

# =============================================================================
# Backend: cpu (pure C, no deps)
# =============================================================================
cpu: CFLAGS = $(CFLAGS_BASE) -DCPU_BUILD
cpu: clean $(TARGET)
	@echo ""
	@echo "Built with CPU backend (pure C)"

# =============================================================================
# Backend: cpu-opt (pure C + OpenMP)
# =============================================================================
cpu-opt: CFLAGS = $(CFLAGS_BASE) -DCPU_BUILD -fopenmp
cpu-opt: LDFLAGS += -fopenmp
cpu-opt: clean $(TARGET)
	@echo ""
	@echo "Built with CPU optimized backend (OpenMP)"

# =============================================================================
# Backend: BLAS (OpenBLAS)
# =============================================================================
blas: CFLAGS = $(CFLAGS_BASE) -DPTTS_USE_BLAS
blas: LDFLAGS = -lm $(BLAS_LIBS)
blas: clean $(TARGET)
	@echo ""
	@echo "Built with BLAS backend (OpenBLAS)"

# =============================================================================
# Backend: CUDA (cuBLAS)
# =============================================================================
cuda: CFLAGS = $(CFLAGS_BASE) -DPTTS_USE_CUDA
cuda: LDFLAGS = -lm $(CUDA_LIBS)
cuda: clean $(CUDA_OBJS) main.o
	$(CC) $(CFLAGS) -o $(TARGET) $(CUDA_OBJS) main.o $(LDFLAGS)
	@echo ""
	@echo "Built with CUDA backend (cuBLAS)"

# =============================================================================
# Backend: CUDA validate (cuBLAS + layer-by-layer validator)
# =============================================================================
cuda-validate: CFLAGS = $(CFLAGS_BASE) -DPTTS_USE_CUDA -DPTTS_CUDA_VALIDATE
cuda-validate: LDFLAGS = -lm $(CUDA_LIBS)
cuda-validate: clean $(CUDA_OBJS) main.o
	$(CC) $(CFLAGS) -o $(TARGET) $(CUDA_OBJS) main.o $(LDFLAGS)
	@echo ""
	@echo "Built with CUDA backend (cuBLAS + PTTS_CUDA_VALIDATE)"

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

ptts_cuda.o: ptts_cuda.c ptts_cuda.h

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c ptts.h ptts_safetensors.h ptts_audio.h ptts_spm.h ptts_flowlm.h ptts_mimi.h ptts_internal.h ptts_kernels.h ptts_cuda.h
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c ptts.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) ptts_cuda.o main.o $(TARGET) $(LIB)

info:
	@echo "Compiler: $(CC)"
	@echo "CFLAGS:   $(CFLAGS_BASE)"

test: cpu
	@PY=python3; \
	if [ -x ./env-syss/bin/python ]; then PY=./env-syss/bin/python; fi; \
	REF=../pocket-tts-hello-world.wav; \
	if [ -f /home/taf2/work/pocket-tts-c/pocket-tts-hello-world.wav ]; then REF=/home/taf2/work/pocket-tts-c/pocket-tts-hello-world.wav; fi; \
	$$PY tools/hello_world_test.py --ptts ./ptts --model-dir ./pocket-tts-model --ref $$REF --frames 17 --seed 123

cuda-validate-test: cuda-validate
	@PTTS_CUDA_VALIDATE=1 PTTS_CUDA_ATTENTION=1 PTTS_CUDA_ATTN_MIN_T=0 ./ptts -d ./pocket-tts-model -p "Hello world. This is a test." -o /tmp/ptts-validate.wav --voice alba --frames 20 --eos-min-frames 20 --seed 123 >/tmp/ptts-validate.log 2>&1; \
	echo "Wrote /tmp/ptts-validate.log"; \
	TH=1e-3; \
	grep -E "CUDA validate" /tmp/ptts-validate.log || true; \
	awk -v th=$$TH '/CUDA validate/ { if ($$NF+0 > th) { bad=1; } } END { if (bad) { print "CUDA validate: FAIL (maxdiff > " th ")"; exit 1 } else { print "CUDA validate: PASS (maxdiff <= " th ")"; } }' /tmp/ptts-validate.log

cuda-sanitize: cuda
	@compute-sanitizer --tool memcheck ./ptts -d ./pocket-tts-model -p "Hello world. This is a test. Can you hear me okay? Great, Let's get started then." -o /tmp/ptts-sanitize.wav --voice alba

# =============================================================================
# Dependencies
# =============================================================================
ptts.o: ptts.c ptts.h ptts_internal.h ptts_safetensors.h ptts_audio.h ptts_spm.h
ptts_audio.o: ptts_audio.c ptts_audio.h
ptts_safetensors.o: ptts_safetensors.c ptts_safetensors.h
ptts_spm.o: ptts_spm.c ptts_spm.h
ptts_kernels.o: ptts_kernels.c ptts_kernels.h
ptts_flowlm.o: ptts_flowlm.c ptts_flowlm.h ptts_internal.h ptts_safetensors.h
ptts_mimi.o: ptts_mimi.c ptts_mimi.h ptts_internal.h ptts_safetensors.h
