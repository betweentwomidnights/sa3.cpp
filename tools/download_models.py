#!/usr/bin/env python3
"""Download the sa3.cpp GGUF model set from HuggingFace into ./models.

Fetches one model variant (DiT + SAME + conditioner) plus the shared T5Gemma encoder
+ tokenizer, following the naming/repo layout in docs/DISTRIBUTION.md. Cross-platform
(Windows + macOS/Linux), unlike a bash models.sh.

  python3 -m pip install -U "huggingface_hub"
  python3 tools/download_models.py --variant medium --encoding f16
  python3 tools/download_models.py --variant medium --encoding f16 --training-base
  HF_TOKEN=hf_... python3 tools/download_models.py --variant small-sfx   # if a repo is gated

For the fastest official Hugging Face path, install a recent `huggingface_hub`
(1.x installs `hf_xet`) and optionally set HF_XET_HIGH_PERFORMANCE=1.

The published default namespace is ``thepatch``; override it with --namespace.
"""
import argparse, importlib.util, os, sys

from model_artifacts import VARIANTS, build_download_plan

DEFAULT_NAMESPACE = "thepatch"

def print_transfer_info():
    try:
        import huggingface_hub
        from huggingface_hub import constants
    except Exception:
        return

    xet_installed = importlib.util.find_spec("hf_xet") is not None
    xet_disabled = bool(getattr(constants, "HF_HUB_DISABLE_XET", False))
    xet_ready = xet_installed and not xet_disabled
    xet_high_perf = bool(getattr(constants, "HF_XET_HIGH_PERFORMANCE", False))
    transfer = "hf_xet" if xet_ready else "http"
    print(f"[hf] huggingface_hub {huggingface_hub.__version__}; transfer={transfer}; "
          f"HF_XET_HIGH_PERFORMANCE={'1' if xet_high_perf else '0'}")
    if xet_disabled:
        print("[hf] HF_HUB_DISABLE_XET=1, so Xet downloads are disabled.")
    elif not xet_installed:
        print('[hf] install "huggingface_hub[hf_xet]" or "hf_xet" for Xet-backed downloads.')
    elif not xet_high_perf:
        print("[hf] set HF_XET_HIGH_PERFORMANCE=1 for Hugging Face's high-throughput Xet mode.")


def main():
    ap = argparse.ArgumentParser(description="Download sa3.cpp GGUF models from HuggingFace.")
    ap.add_argument("--variant", default="medium", choices=list(VARIANTS))
    ap.add_argument("--encoding", default="f16", choices=["f16", "f32"],
                    help="weight encoding for DiT + SAME (default f16, the production path)")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="HuggingFace org/user")
    ap.add_argument("--out", default="models", help="output dir (default ./models)")
    ap.add_argument("--training-base", action="store_true",
                    help="also fetch the matching -base DiT required for LoRA training")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import get_token
    except ImportError:
        sys.exit('missing dependency: python3 -m pip install -U "huggingface_hub"')

    plan = build_download_plan(args.namespace, args.variant, args.encoding, args.training_base)

    os.makedirs(args.out, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or get_token()
    print_transfer_info()
    for repo, files in plan:
        print(f"[download] {repo}")
        for fname in files:
            print(f"           {fname}")
        try:
            snapshot_download(
                repo_id=repo,
                allow_patterns=files,
                local_dir=args.out,
                max_workers=min(8, len(files)),
                token=token,
            )
        except Exception as e:
            print(f"\n[error] could not download {repo}", file=sys.stderr)
            print("        If the repo is private, make sure the active token can read that namespace.", file=sys.stderr)
            print("        Fine-grained Hugging Face tokens must include repo.content.read for the org/user that owns the repo.", file=sys.stderr)
            raise
    suffix = " + training base" if args.training_base else ""
    print(f"[done] {args.variant} ({args.encoding}){suffix} -> {args.out}/")
    print("run: sa3-generate --tok <vocab> --t5 <encoder> --cond <conditioner> "
          "--dit <dit> --same <same> --prompt \"...\" --out song.wav")


if __name__ == "__main__":
    main()
