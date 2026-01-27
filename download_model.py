#!/usr/bin/env python3
"""Download Pocket-TTS weights into a local directory."""

import argparse
import os
import sys

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError
except Exception:  # pragma: no cover
    print("error: huggingface_hub is required (pip install huggingface_hub)")
    sys.exit(1)

DEFAULT_REV_FULL = "427e3d61b276ed69fdd03de0d185fa8a8d97fc5b"
DEFAULT_REV_NOVOICE = "d4fdd22ae8c8e1cb3634e150ebeff1dab2d16df3"


def download(repo_id: str, filename: str, revision: str, out_dir: str) -> str:
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
        )
        return path
    except GatedRepoError:
        print("")
        print("error: access to this Hugging Face repo is gated.")
        print(f"repo: {repo_id}")
        print("Please visit the model page, accept the terms, and authenticate.")
        print("Then rerun with one of:")
        print("  - huggingface-cli login")
        print("  - export HF_TOKEN=hf_...your_token...")
        print("")
        print("Model page:")
        print("  https://huggingface.co/kyutai/pocket-tts")
        sys.exit(1)
    except Exception as exc:
        print(f"error: download failed: {exc}")
        sys.exit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="pocket-tts-model", help="Output directory")
    parser.add_argument(
        "--without-voice",
        action="store_true",
        help="Use the no-voice-cloning repo for weights",
    )
    parser.add_argument(
        "--voice",
        default="alba",
        help="Download a voice embedding by name (default: alba, use 'none' to skip)",
    )
    parser.add_argument(
        "--no-voice-embed",
        action="store_true",
        help="Skip downloading voice embeddings",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override weights revision (defaults to config-pinned hash)",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.without_voice:
        weights_repo = "kyutai/pocket-tts-without-voice-cloning"
        weights_rev = args.revision or DEFAULT_REV_NOVOICE
    else:
        weights_repo = "kyutai/pocket-tts"
        weights_rev = args.revision or DEFAULT_REV_FULL

    tokenizer_repo = "kyutai/pocket-tts-without-voice-cloning"
    tokenizer_rev = DEFAULT_REV_NOVOICE

    print(f"Downloading weights from {weights_repo}@{weights_rev}...")
    download(weights_repo, "tts_b6369a24.safetensors", weights_rev, args.out)

    print(f"Downloading tokenizer from {tokenizer_repo}@{tokenizer_rev}...")
    download(tokenizer_repo, "tokenizer.model", tokenizer_rev, args.out)

    if not args.no_voice_embed:
        voice_name = (args.voice or "").strip()
        if voice_name and voice_name not in ("none", "off", "null"):
            voice_repo = "kyutai/pocket-tts-without-voice-cloning"
            voice_rev = DEFAULT_REV_NOVOICE
            voice_file = f"embeddings/{voice_name}.safetensors"
            print(f"Downloading voice embedding {voice_name} from {voice_repo}@{voice_rev}...")
            download(voice_repo, voice_file, voice_rev, args.out)

    print(f"Done. Files saved to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
