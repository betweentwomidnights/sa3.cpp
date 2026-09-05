#!/usr/bin/env python3
"""Download the sa3.cpp GGUF model set from HuggingFace into ./models.

By default, fetches one SA3 model variant. ``--sat`` selects an optional
stable-audio-tools family instead; currently that means SAOS and its validated finetunes.

  python3 -m pip install -U "huggingface_hub"
  python3 tools/download_models.py --variant medium --encoding f16
  python3 tools/download_models.py --variant medium --encoding q4_k_m --training-base
  python3 tools/download_models.py --variant medium --encoding q8_0 --dry-run
  python3 tools/download_models.py --sat --sat-model saos --saos-variant jerry-grunge
  HF_TOKEN=hf_... python3 tools/download_models.py --variant small-sfx   # if a repo is gated

For the fastest official Hugging Face path, install a recent `huggingface_hub`
(1.x installs `hf_xet`) and optionally set HF_XET_HIGH_PERFORMANCE=1.

The published default namespace is ``thepatch``; override it with --namespace.
"""
import argparse, importlib.util, os, sys

from model_artifacts import (ENCODINGS, SAOS_DEFAULT_ENCODING, SAOS_ENCODINGS, SAOS_VARIANTS,
                             TEXT_ENCODER_DEFAULT, TEXT_ENCODER_ENCODINGS, VARIANTS,
                             build_download_plan, build_saos_download_plan)

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
    ap.add_argument("--sat", action="store_true",
                    help="download an optional stable-audio-tools family instead of SA3")
    ap.add_argument("--sat-model", default="saos", choices=["saos"],
                    help="stable-audio-tools architecture family (currently: saos)")
    ap.add_argument("--saos-variant", default="arc", choices=list(SAOS_VARIANTS),
                    help="SAOS checkpoint: arc, kickbass, or jerry-grunge")
    ap.add_argument("--variant", default="medium", choices=list(VARIANTS))
    all_encodings = sorted({e.lower() for e in ENCODINGS + SAOS_ENCODINGS})
    ap.add_argument("--encoding", default=None, choices=all_encodings,
                    help="DiT encoding (default: SA3 f16; SAOS q5_k_m)")
    ap.add_argument("--ae-encoding", dest="ae_encoding", default=None,
                    choices=[e.lower() for e in ENCODINGS],
                    help="autoencoder encoding, on its own axis from --encoding "
                         "(default f32; it used to follow --encoding, so a quantized DiT "
                         "silently fetched a quantized SAME)")
    ap.add_argument("--t5-encoding", default=None, choices=all_encodings,
                    help="text encoder encoding (default: SA3 f16; SAOS follows --encoding)")
    ap.add_argument("--namespace", default=DEFAULT_NAMESPACE, help="HuggingFace org/user")
    ap.add_argument("--out", default="models", help="output dir (default ./models)")
    ap.add_argument("--training-base", action="store_true",
                    help="also fetch the matching -base DiT required for LoRA training")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved repo/file plan and exit, downloading nothing")
    args = ap.parse_args()

    if args.sat:
        if args.training_base:
            ap.error("--training-base applies to SA3, not --sat")
        encoding = args.encoding or SAOS_DEFAULT_ENCODING.lower()
        if encoding.upper() not in SAOS_ENCODINGS:
            ap.error(f"SAOS does not publish {encoding}; choose f16, q8_0, q5_k_m, or q4_k_m")
        for option, value in (("--t5-encoding", args.t5_encoding),
                              ("--ae-encoding", args.ae_encoding)):
            if value and value.upper() not in SAOS_ENCODINGS:
                ap.error(f"SAOS {option} does not publish {value}")
        plan = build_saos_download_plan(
            args.namespace, args.saos_variant, encoding,
            text_encoding=args.t5_encoding, ae_encoding=args.ae_encoding,
        )
    else:
        encoding = args.encoding or "f16"
        text_encoding = args.t5_encoding or TEXT_ENCODER_DEFAULT.lower()
        if text_encoding.upper() not in TEXT_ENCODER_ENCODINGS:
            ap.error("SA3 --t5-encoding must be f16, f32, or q8_0")
        plan = build_download_plan(args.namespace, args.variant, encoding, args.training_base,
                                   text_encoding=text_encoding, ae_encoding=args.ae_encoding)

    if args.dry_run:
        for repo, files in plan:
            for fname in files:
                print(f"[plan] https://huggingface.co/{repo}/resolve/main/{fname} -> "
                      f"{os.path.join(args.out, fname)}")
        return

    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import get_token
    except ImportError:
        sys.exit('missing dependency: python3 -m pip install -U "huggingface_hub"')

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
    if args.sat:
        print(f"[done] SAOS {args.saos_variant} ({encoding}) -> {args.out}/")
        print(f"run: saos-generate --model {args.saos_variant} --models-dir {args.out} "
              "--prompt \"...\" --out audio.wav")
    else:
        suffix = " + training base" if args.training_base else ""
        print(f"[done] {args.variant} ({encoding}){suffix} -> {args.out}/")
        print("run: sa3-generate --tok <vocab> --t5 <encoder> --cond <conditioner> "
              "--dit <dit> --same <same> --prompt \"...\" --out song.wav")


if __name__ == "__main__":
    main()
