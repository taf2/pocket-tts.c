# Implementation Notes (WIP)

Goal: a small, dependency-free C port of Pocket-TTS, in the spirit of `flux2.c`.

## Pipeline (high level)

1. **Text conditioning**
   - Prompt preparation (capitalize, punctuation, padding)
   - SentencePiece tokenizer
   - LUT embedding for tokens

2. **Voice conditioning**
   - Precomputed `audio_prompt` embeddings (1024-dim tokens)
   - Loaded from `embeddings/<voice>.safetensors` (default: alba)

3. **FlowLM**
   - Flow matching transformer
   - Generates acoustic latents

4. **Mimi codec**
   - Decode latents to waveform
   - 24kHz mono output

5. **Streaming**
   - Stateful modules for chunked generation
   - ~80ms per frame (12.5 Hz)

## Model assets

Pinned in `pocket-tts/pocket_tts/config/b6369a24.yaml`:
- Weights: `tts_b6369a24.safetensors`
- Tokenizer: `tokenizer.model`
- Voice embeddings: `embeddings/<voice>.safetensors` (e.g., `alba`)

## Open tasks

- Validate SentencePiece normalization parity against reference outputs
- Safetensors -> parameter loader for FlowLM and Mimi
- Transformer kernels (matmul, softmax, RMSNorm, etc.)
- Mimi decoder implementation
- Streaming API and CLI progress output

## Current debug hooks

- `./ptts --flow-test` runs a single non-streaming FlowLM step and prints latent stats.
- `./ptts --mimi-test` runs FlowLM + Mimi decoder-transformer stats.
- `./ptts --mimi-wave` decodes FlowLM latents through Mimi into a short WAV
  (`--frames N` gives N * 80ms, auto frames + EOS stop by default).
