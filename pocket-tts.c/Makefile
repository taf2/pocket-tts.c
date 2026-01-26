# Pocket-TTS Pure C (WIP)
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm

SRCS = ptts.c ptts_audio.c ptts_safetensors.c ptts_spm.c ptts_flowlm.c ptts_mimi.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = ptts
LIB = libptts.a

.PHONY: all clean help generic lib info test

all: help

help:
	@echo "Pocket-TTS Pure C (WIP) - Build Targets"
	@echo ""
	@echo "  make generic  - Pure C, no dependencies"
	@echo ""
	@echo "Other targets:"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo "  make lib      - Build static library"
	@echo "  make test     - Run hello world regression test"
	@echo ""
	@echo "Example: make generic && ./ptts --dummy -p \"hello\" -o out.wav"

# =============================================================================
# Backend: generic (pure C, no deps)
# =============================================================================
generic: CFLAGS = $(CFLAGS_BASE) -DGENERIC_BUILD
generic: clean $(TARGET)
	@echo ""
	@echo "Built with GENERIC backend (pure C)"

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

lib: $(LIB)

$(LIB): $(OBJS)
	ar rcs $@ $^

%.o: %.c ptts.h ptts_safetensors.h ptts_audio.h ptts_spm.h ptts_flowlm.h ptts_mimi.h ptts_internal.h
	$(CC) $(CFLAGS) -c -o $@ $<

main.o: main.c ptts.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) main.o $(TARGET) $(LIB)

info:
	@echo "Compiler: $(CC)"
	@echo "CFLAGS:   $(CFLAGS_BASE)"

test: generic
	@PY=python3; \
	if [ -x ./env-syss/bin/python ]; then PY=./env-syss/bin/python; fi; \
	REF=../pocket-tts-hello-world.wav; \
	if [ -f /home/taf2/work/pocket-tts-c/pocket-tts-hello-world.wav ]; then REF=/home/taf2/work/pocket-tts-c/pocket-tts-hello-world.wav; fi; \
	$$PY tools/hello_world_test.py --ptts ./ptts --model-dir ./pocket-tts-model --ref $$REF --frames 17

# =============================================================================
# Dependencies
# =============================================================================
ptts.o: ptts.c ptts.h ptts_internal.h ptts_safetensors.h ptts_audio.h ptts_spm.h
ptts_audio.o: ptts_audio.c ptts_audio.h
ptts_safetensors.o: ptts_safetensors.c ptts_safetensors.h
ptts_spm.o: ptts_spm.c ptts_spm.h
ptts_flowlm.o: ptts_flowlm.c ptts_flowlm.h ptts_internal.h ptts_safetensors.h
ptts_mimi.o: ptts_mimi.c ptts_mimi.h ptts_internal.h ptts_safetensors.h
