# Pocket-TTS Pure C (WIP)

A minimal, dependency-free C scaffold for a Pocket-TTS port, following the style of `flux2.c`.
The goal is a small, pure-C TTS engine that loads Pocket-TTS weights and runs on CPU.

**Status:** end-to-end TTS works (FlowLM + Mimi). CUDA path is validated and fast. CLI and tests are
kept small in the `flux2.c` spirit.

## Quick Start

```bash
# Build
make cpu
# or: make blas  # OpenBLAS accelerated
# or: make cuda  # NVIDIA CUDA + cuBLAS + NVRTC

CUDA builds require the CUDA toolkit + cuBLAS + NVRTC.

# Download model weights + default voice embedding (see note about HF access below)
./download_model.sh --voice alba

# List tensors in the weights file
./ptts -d pocket-tts-model --list

# Generate speech
./ptts -d pocket-tts-model -p "Hello world" -o out.wav --voice alba
```

## Sample Audio

<video src="https://github.com/user-attachments/assets/adcf2f2e-ffee-4b0e-a98a-f174bb06b4d9" controls></video>

## Features

- Pure C, no external runtime dependencies
- Safetensors reader (mmap) for model inspection
- Simple WAV writer
- CLI designed like `flux2.c`

## Usage

```bash
./ptts -d pocket-tts-model -p "Hello world" -o out.wav [options]
```

Options:
```
-d, --dir PATH        Model directory or .safetensors file
-p, --prompt TEXT     Text to synthesize
-o, --output PATH     Output WAV path
    --info            Print model info
    --list            List tensors
    --find TEXT       List tensors whose names contain TEXT
    --verify          Verify weights against expected shapes
    --tokens          Print token IDs for the prompt
    --flow-test       Run a single FlowLM step and print latent stats
    --mimi-test       Run FlowLM + Mimi decoder transformer stats
    --mimi-wave PATH  Write Mimi decode WAV to PATH (frames * 80ms)
    --frames N        Number of FlowLM/Mimi frames (affects --mimi-wave and -o, default: auto)
    --latent-out PATH Write raw FlowLM latents (float32, 32 values per frame) to PATH
    --cond-out PATH   Write first FlowLM condition vector (1024 floats)
    --flow-out PATH   Write first FlowLM flow vector (32 floats)
    --voice NAME      Voice embedding name or .safetensors path (default: alba)
    --dummy           Generate placeholder audio (no model)
-r, --rate N          Sample rate for dummy generator (default: 24000)
-t, --temp F          Noise temperature for FlowLM (default: 1.0)
    --noise-clamp F   Clamp noise to [-F, F] (default: 0, off)
    --eos-threshold F Stop early if eos_logit >= F (default: -4.0)
    --eos-min-frames N Minimum frames before EOS stop (default: 1)
    --eos-after N    Frames to keep after EOS (default: auto)
-q, --quiet           Less output
-v, --verbose         More output
-h, --help            Show help
```

## Library API (WIP)

```c
#include "ptts.h"

ptts_ctx *ctx = ptts_load_dir("pocket-tts-model");
if (!ctx) {
    fprintf(stderr, "error: %s\n", ptts_get_error());
    return 1;
}

ptts_params params = PTTS_PARAMS_DEFAULT;
ptts_audio *audio = ptts_generate(ctx, "Hello world", "alba", &params);
if (!audio) {
    fprintf(stderr, "generate failed: %s\n", ptts_get_error());
    ptts_free(ctx);
    return 1;
}

ptts_audio_save_wav(audio, "out.wav");
ptts_audio_free(audio);
ptts_free(ctx);
```

`ptts_generate()` runs FlowLM + Mimi with auto frame estimation + EOS stop.

## Parity check (FlowLM)

There is a small helper to compare C latents against the Python reference:

```bash
python3 tools/flowlm_parity.py --text "Hello world" --frames 1 --temp 0
```

## Tests (Golden Regression)

`make test` runs a deterministic “Hello world!” golden test against a reference WAV.
Set `PTTS_HELLO_REF` if your reference file lives elsewhere.

```bash
make test
# or:
PTTS_HELLO_REF=/path/to/hello.wav make test
```

## Model Download Notes

Pocket-TTS weights are hosted on Hugging Face and may require accepting model terms.
`download_model.py` uses `huggingface_hub` and will respect `HF_TOKEN` if set.

Voice embeddings (e.g., `embeddings/alba.safetensors`) are required for conditioning.
Use `./download_model.sh --voice alba` or pass `--voice none` to disable conditioning.

## Build

```bash
make cpu
# or: make blas
# or: make cuda
# or: make mps
```

### Apple Silicon (MPS)

For accelerated inference on Apple Silicon Macs:

```bash
make mps
./ptts -d model -p "Hello world" -o out.wav
```

Requirements:
- macOS 12.0+ (Monterey or later)
- Apple Silicon (M1/M2/M3) or AMD GPU with Metal support

CUDA diagnostics:

```bash
make cuda-validate-test
make cuda-sanitize
```

## Roadmap (high level)

- SentencePiece tokenizer in C
- FlowLM forward pass
- Mimi codec decoder
- Streaming API (chunked audio)

## License

MIT
