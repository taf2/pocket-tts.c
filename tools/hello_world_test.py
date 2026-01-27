#!/usr/bin/env python3
import argparse
import math
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        n = w.getnframes()
        data = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
    data /= 32768.0
    return sr, data


def align_signals(a: np.ndarray, b: np.ndarray, max_lag: int):
    min_len = min(len(a), len(b))
    if min_len < 1000:
        return a[:min_len], b[:min_len], 0, 0.0

    best_lag = 0
    best_corr = -1e9
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            xa = a[-lag:min_len]
            xb = b[: min_len + lag]
        elif lag > 0:
            xa = a[: min_len - lag]
            xb = b[lag:min_len]
        else:
            xa = a[:min_len]
            xb = b[:min_len]
        if len(xa) < 1000:
            continue
        corr = float(np.dot(xa, xb))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    if best_lag < 0:
        xa = a[-best_lag:min_len]
        xb = b[: min_len + best_lag]
    elif best_lag > 0:
        xa = a[: min_len - best_lag]
        xb = b[best_lag:min_len]
    else:
        xa = a[:min_len]
        xb = b[:min_len]

    denom = float(np.linalg.norm(xa) * np.linalg.norm(xb) + 1e-12)
    corr_norm = float(np.dot(xa, xb) / denom)
    return xa, xb, best_lag, corr_norm


def log_mag_mse(a: np.ndarray, b: np.ndarray, n_fft: int = 1024, hop: int = 256):
    win = np.hanning(n_fft).astype(np.float32)
    max_frames = min((len(a) - n_fft) // hop, (len(b) - n_fft) // hop)
    if max_frames <= 0:
        return float("inf"), 0
    total = 0.0
    for i in range(max_frames):
        sa = a[i * hop : i * hop + n_fft] * win
        sb = b[i * hop : i * hop + n_fft] * win
        ma = np.abs(np.fft.rfft(sa))
        mb = np.abs(np.fft.rfft(sb))
        la = np.log1p(ma)
        lb = np.log1p(mb)
        diff = la - lb
        total += float(np.mean(diff * diff))
    return total / max_frames, max_frames


def generate_audio(ptts: str, model_dir: str, prompt: str, voice: str,
                   temp: float, seed: int, frames: int | None) -> str:
    fd, out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    cmd = [ptts, "-d", model_dir, "-p", prompt, "-o", out_path, "--voice", voice,
           "-t", str(temp), "-S", str(seed), "-q"]
    if frames is not None:
        cmd += ["--frames", str(frames)]
    subprocess.run(cmd, check=True)
    return out_path


def main() -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_ref = os.environ.get(
        "PTTS_HELLO_REF",
        os.path.abspath(os.path.join(root, "..", "pocket-tts-hello-world.wav")),
    )

    parser = argparse.ArgumentParser(description="Hello world regression test")
    parser.add_argument("--ptts", default=os.path.join(root, "ptts"), help="Path to ptts binary")
    parser.add_argument("--model-dir", default=os.path.join(root, "pocket-tts-model"), help="Model dir")
    parser.add_argument("--prompt", default="Hello world!", help="Prompt text")
    parser.add_argument("--voice", default="alba", help="Voice embedding name/path")
    parser.add_argument("--ref", default=default_ref, help="Reference WAV path")
    parser.add_argument("--gen", default=None, help="Generated WAV path (skip generation)")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--frames", type=int, default=None, help="Fixed frame count")
    parser.add_argument("--max-lag", type=int, default=2000, help="Max alignment lag in samples")
    parser.add_argument("--max-duration-diff", type=float, default=0.5, help="Max duration diff (sec)")
    parser.add_argument("--min-corr", type=float, default=0.0, help="Min normalized correlation")
    parser.add_argument("--max-logmag-mse", type=float, default=1.0, help="Max log-magnitude MSE")
    parser.add_argument("--min-rms-ratio", type=float, default=0.3, help="Min RMS ratio")
    parser.add_argument("--max-rms-ratio", type=float, default=3.0, help="Max RMS ratio")
    parser.add_argument("--min-peak-ratio", type=float, default=0.3, help="Min peak ratio")
    parser.add_argument("--max-peak-ratio", type=float, default=3.0, help="Max peak ratio")
    args = parser.parse_args()

    if not os.path.exists(args.ref):
        print(f"error: reference wav not found: {args.ref}")
        return 2

    gen_path = args.gen
    temp_path = None
    if gen_path is None:
        temp_path = generate_audio(args.ptts, args.model_dir, args.prompt, args.voice,
                                   args.temp, args.seed, args.frames)
        gen_path = temp_path

    sr_ref, ref = read_wav(args.ref)
    sr_gen, gen = read_wav(gen_path)
    if sr_ref != sr_gen:
        print(f"error: sample rate mismatch ref={sr_ref} gen={sr_gen}")
        return 2

    dur_ref = len(ref) / sr_ref
    dur_gen = len(gen) / sr_gen
    dur_diff = abs(dur_ref - dur_gen)

    ref_aligned, gen_aligned, lag, corr = align_signals(ref, gen, args.max_lag)
    log_mse, n_frames = log_mag_mse(ref_aligned, gen_aligned)

    rms_ref = math.sqrt(float(np.mean(ref_aligned * ref_aligned)))
    rms_gen = math.sqrt(float(np.mean(gen_aligned * gen_aligned)))
    peak_ref = float(np.max(np.abs(ref_aligned)))
    peak_gen = float(np.max(np.abs(gen_aligned)))

    rms_ratio = rms_gen / (rms_ref + 1e-12)
    peak_ratio = peak_gen / (peak_ref + 1e-12)

    print("Hello world test report")
    print(f"  ref: {args.ref}")
    print(f"  gen: {gen_path}")
    print(f"  duration: ref={dur_ref:.3f}s gen={dur_gen:.3f}s diff={dur_diff:.3f}s")
    print(f"  align: lag={lag} corr={corr:.3f}")
    print(f"  log-magnitude MSE: {log_mse:.4f} (frames={n_frames})")
    print(f"  rms ratio: {rms_ratio:.3f} peak ratio: {peak_ratio:.3f}")

    ok = True
    if dur_diff > args.max_duration_diff:
        print("  FAIL: duration diff too large")
        ok = False
    if corr < args.min_corr:
        print("  FAIL: correlation too low")
        ok = False
    if log_mse > args.max_logmag_mse:
        print("  FAIL: log-magnitude MSE too high")
        ok = False
    if not (args.min_rms_ratio <= rms_ratio <= args.max_rms_ratio):
        print("  FAIL: RMS ratio out of bounds")
        ok = False
    if not (args.min_peak_ratio <= peak_ratio <= args.max_peak_ratio):
        print("  FAIL: peak ratio out of bounds")
        ok = False

    if temp_path:
        os.unlink(temp_path)

    if ok:
        print("  PASS")
        return 0
    print("  FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
