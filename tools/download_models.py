#!/usr/bin/env python3
"""Download the sa3.cpp GGUF model set from HuggingFace into ./models.

Fetches one model variant (DiT + SAME + conditioner) plus the shared T5Gemma encoder
+ tokenizer, following the naming/repo layout in docs/DISTRIBUTION.md. Cross-platform
(Windows + macOS/Linux), unlike a bash models.sh.

  python3 -m pip install huggingface_hub
  python3 tools/download_models.py --variant medium --encoding f16
  HF_TOKEN=hf_... python3 tools/download_models.py --variant small-sfx   # if a repo is gated

The HF namespace is not final yet (pending Stability packaging) — override with --namespace.
"""
import argparse, os, sys

# TODO: confirm with Stability/Zach before publishing; override at runtime with --namespace.
DEFAULT_NAMESPACE = "thepatch"

# variant -> (dit size_label, SAME suffix)
VARIANTS = {
    "medium":      ("1.5B", "same-l"),
    "small-music": ("0.5B", "same-s"),
    "small-sfx":   ("0.5B", "same-s"),
}
SHARED_REPO = "t5gemma-b-b-ul2-GGUF"


def main():
    ap = argparse.ArgumentParser(description="Download sa3.cpp GGUF models from HuggingFace.")
    ap.add_argument("--variant", default="medium", choices=list(VARIANTS))
    ap.add_argument("--encoding", default="f16", choices=["f16", "f32"],
                    help="weight encoding for DiT + SAME (default f16, the production path)")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="HuggingFace org/user")
    ap.add_argument("--out", default="models", help="output dir (default ./models)")
    args = ap.parse_args()

    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import get_token
    except ImportError:
        sys.exit("missing dependency: python3 -m pip install huggingface_hub")

    enc = args.encoding.upper()                 # F16 / F32
    dit_size, same = VARIANTS[args.variant]
    var_repo = f"{args.namespace}/stable-audio-3-{args.variant}-GGUF"
    shared   = f"{args.namespace}/{SHARED_REPO}"
    base     = f"stable-audio-3-{args.variant}"

    # (repo, filename). conditioner/encoder/tokenizer are small + quality-critical -> always F32.
    wanted = [
        (var_repo, f"{base}-dit-{dit_size}-v1.0-{enc}.gguf"),
        (var_repo, f"{base}-{same}-v1.0-{enc}.gguf"),
        (var_repo, f"{base}-conditioner-v1.0-F32.gguf"),
        (shared,   "t5gemma-b-b-ul2-encoder-0.3B-v1.0-F32.gguf"),
        (shared,   "t5gemma-b-b-ul2-v1.0-vocab.gguf"),
    ]

    os.makedirs(args.out, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or get_token()
    for repo, fname in wanted:
        print(f"[download] {repo}/{fname}")
        try:
            hf_hub_download(repo_id=repo, filename=fname, local_dir=args.out, token=token)
        except Exception as e:
            print(f"\n[error] could not download {repo}/{fname}", file=sys.stderr)
            print("        If the repo is private, make sure the active token can read that namespace.", file=sys.stderr)
            print("        Fine-grained Hugging Face tokens must include repo.content.read for the org/user that owns the repo.", file=sys.stderr)
            raise
    print(f"[done] {args.variant} ({args.encoding}) -> {args.out}/")
    print("run: sa3-generate --tok <vocab> --t5 <encoder> --cond <conditioner> "
          "--dit <dit> --same <same> --prompt \"...\" --out song.wav")


if __name__ == "__main__":
    main()
