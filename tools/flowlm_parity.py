#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def add_pocket_tts_to_path(root: Path) -> Path:
    repo = root.parent / "pocket-tts"
    sys.path.insert(0, str(repo))
    return repo


def resolve_voice_path(model_dir: Path, voice: str) -> Path | None:
    voice = voice.strip()
    if voice in ("none", "off", "null", ""):
        return None
    candidate = Path(voice)
    if candidate.is_file():
        return candidate
    if "/" in voice or voice.endswith(".safetensors"):
        candidate = model_dir / voice
        if candidate.is_file():
            return candidate
    candidate = model_dir / "embeddings" / f"{voice}.safetensors"
    if candidate.is_file():
        return candidate
    candidate = model_dir / "voices" / f"{voice}.safetensors"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Voice embedding not found for '{voice}'")


def run_python_ref(weights_path: Path, config_path: Path, model_dir: Path, voice: str,
                   text: str, frames: int, steps: int, temp: float,
                   noise_clamp: float, eos_threshold: float,
                   seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch
    from pocket_tts.models.tts_model import TTSModel, prepare_text_prompt
    from pocket_tts.modules.stateful_module import init_states
    from pocket_tts.utils.config import load_config
    from safetensors.torch import load_file

    cfg = load_config(config_path)
    cfg.weights_path = str(weights_path)
    cfg.weights_path_without_voice_cloning = str(weights_path)
    model = TTSModel._from_pydantic_config_with_weights(
        cfg, temp, steps, None if noise_clamp <= 0 else noise_clamp, eos_threshold
    )
    model.eval()

    torch.manual_seed(seed)

    text, _ = prepare_text_prompt(text)
    prepared = model.flow_lm.conditioner.prepare(text)

    model_state = init_states(model.flow_lm, batch_size=1, sequence_length=1000)
    voice_path = resolve_voice_path(model_dir, voice)
    if voice_path:
        prompt = load_file(str(voice_path))["audio_prompt"].to(model.flow_lm.device)
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0)
        _ = model._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)
    _ = model._run_flow_lm_and_increment_step(model_state=model_state, text_tokens=prepared.tokens)

    def clone_state(state: dict) -> dict:
        out = {}
        for mk, mv in state.items():
            out[mk] = {sk: sv.clone() for sk, sv in mv.items()}
        return out

    state_for_cond = clone_state(model_state)
    empty_text = torch.empty((1, 0, model.flow_lm.dim), dtype=model.flow_lm.dtype)
    bos_seq = torch.full((1, 1, model.flow_lm.ldim), fill_value=float("NaN"), dtype=model.flow_lm.dtype)
    bos_seq = torch.where(torch.isnan(bos_seq), model.flow_lm.bos_emb, bos_seq)
    inp = model.flow_lm.input_linear(bos_seq)
    transformer_out = model.flow_lm.backbone(inp, empty_text, bos_seq, model_state=state_for_cond)
    transformer_out = transformer_out.to(torch.float32)
    transformer_out = transformer_out[:, -1]
    cond = transformer_out[0].detach().cpu().numpy().astype(np.float32)
    s = torch.zeros((1, 1), dtype=transformer_out.dtype)
    t = torch.full((1, 1), 1.0 / max(steps, 1), dtype=transformer_out.dtype)
    x0 = torch.zeros((1, model.flow_lm.ldim), dtype=transformer_out.dtype)
    flow = model.flow_lm.flow_net(transformer_out, s, t, x0)
    flow = flow[0].detach().cpu().numpy().astype(np.float32)

    back_in = torch.full((1, 1, model.flow_lm.ldim), fill_value=float("NaN"), dtype=model.flow_lm.dtype)
    latents = []
    with torch.no_grad():
        for _ in range(frames):
            latent, _is_eos = model._run_flow_lm_and_increment_step(
                model_state=model_state, backbone_input_latents=back_in
            )
            latents.append(latent[0, 0].cpu().numpy().astype(np.float32))
            back_in = latent
    return np.stack(latents, axis=0), cond, flow


def run_c_ref(ptts_path: Path, model_dir: Path, voice: str, text: str, frames: int,
              steps: int, temp: float, noise_clamp: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    with tempfile.NamedTemporaryFile(delete=False) as tmpc:
        cond_path = tmpc.name
    with tempfile.NamedTemporaryFile(delete=False) as tmpf:
        flow_path = tmpf.name
    cmd = [
        str(ptts_path),
        "-d", str(model_dir),
        "-p", text,
        "--flow-test",
        "--voice", voice,
        "--frames", str(frames),
        "--latent-out", tmp_path,
        "--cond-out", cond_path,
        "--flow-out", flow_path,
        "-s", str(steps),
        "-t", str(temp),
        "--noise-clamp", str(noise_clamp),
        "-S", str(seed),
        "--eos-threshold", "1e9",
        "--eos-min-frames", "1",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    data = np.fromfile(tmp_path, dtype=np.float32)
    cond = np.fromfile(cond_path, dtype=np.float32)
    flow = np.fromfile(flow_path, dtype=np.float32)
    os.unlink(tmp_path)
    os.unlink(cond_path)
    os.unlink(flow_path)
    if data.size % 32 != 0:
        raise RuntimeError(f"Unexpected latent size: {data.size}")
    if cond.size != 1024:
        raise RuntimeError(f"Unexpected cond size: {cond.size}")
    if flow.size != 32:
        raise RuntimeError(f"Unexpected flow size: {flow.size}")
    return data.reshape(-1, 32), cond, flow


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    add_pocket_tts_to_path(root)

    parser = argparse.ArgumentParser(description="FlowLM parity (Python vs C)")
    parser.add_argument("--text", default="Hello world", help="Prompt text")
    parser.add_argument("--frames", type=int, default=1, help="Number of frames to compare")
    parser.add_argument("--steps", type=int, default=4, help="LSD decode steps")
    parser.add_argument("--temp", type=float, default=0.0, help="Noise temperature (0 for deterministic)")
    parser.add_argument("--noise-clamp", type=float, default=0.0, help="Noise clamp")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--eos-threshold", type=float, default=-2.0, help="EOS threshold (python only)")
    parser.add_argument("--voice", default="none", help="Voice embedding name or path (default: none)")
    parser.add_argument("--ptts", default=str(root / "ptts"), help="Path to ptts binary")
    parser.add_argument("--model-dir", default=str(root / "pocket-tts-model"), help="Model dir for C")
    parser.add_argument("--weights", default=str(root / "pocket-tts-model" / "tts_b6369a24.safetensors"),
                        help="Weights path for Python")
    parser.add_argument("--config", default=str(root.parent / "pocket-tts" / "pocket_tts" / "config" / "b6369a24.yaml"),
                        help="Config path for Python")
    args = parser.parse_args()

    py_latents, py_cond, py_flow = run_python_ref(
        Path(args.weights), Path(args.config), Path(args.model_dir), args.voice, args.text, args.frames,
        args.steps, args.temp, args.noise_clamp, args.eos_threshold, args.seed
    )
    c_latents, c_cond, c_flow = run_c_ref(
        Path(args.ptts), Path(args.model_dir), args.voice, args.text, args.frames,
        args.steps, args.temp, args.noise_clamp, args.seed
    )

    if py_latents.shape != c_latents.shape:
        print(f"Shape mismatch: py={py_latents.shape} c={c_latents.shape}")
        return 1

    diff = py_latents - c_latents
    max_abs = np.max(np.abs(diff))
    mean_abs = np.mean(np.abs(diff))
    rms = np.sqrt(np.mean(diff * diff))
    print("Parity report:")
    print(f"  frames: {py_latents.shape[0]}")
    print(f"  max_abs: {max_abs:.6f}")
    print(f"  mean_abs: {mean_abs:.6f}")
    print(f"  rms: {rms:.6f}")

    cond_diff = py_cond - c_cond
    cmax = np.max(np.abs(cond_diff))
    cmean = np.mean(np.abs(cond_diff))
    crms = np.sqrt(np.mean(cond_diff * cond_diff))
    print("Cond report:")
    print(f"  max_abs: {cmax:.6f}")
    print(f"  mean_abs: {cmean:.6f}")
    print(f"  rms: {crms:.6f}")
    fdiff = py_flow - c_flow
    fmax = np.max(np.abs(fdiff))
    fmean = np.mean(np.abs(fdiff))
    frms = np.sqrt(np.mean(fdiff * fdiff))
    print("Flow report:")
    print(f"  max_abs: {fmax:.6f}")
    print(f"  mean_abs: {fmean:.6f}")
    print(f"  rms: {frms:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
