"""Pure naming and download-manifest helpers for published sa3.cpp artifacts."""

VERSION = "v1.0"

# variant -> (DiT size label, SAME suffix)
VARIANTS = {
    "medium": ("1.5B", "same-l"),
    "small-music": ("0.5B", "same-s"),
    "small-sfx": ("0.5B", "same-s"),
}

UPSTREAM_BASE_REPOS = {
    variant: f"https://huggingface.co/stabilityai/stable-audio-3-{variant}-base"
    for variant in VARIANTS
}

UPSTREAM_BASE_REVISIONS = {
    "medium": "b32993f73c3bdc3864043a72d8032606bba737c8",
    "small-music": "eab5ceee5ad9c1ed38800aff30a8e49d1161c539",
    "small-sfx": "cc5ddb990e30daa68336ac61c140c37c7033ab7c",
}

STABILITY_LICENSE_SHA256 = "d6f6b1a4dce5c852bd6d7d9482d002baf0ccdb71e662250b73be9eec8764ee8d"

SHARED_REPO = "t5gemma-b-b-ul2-GGUF"

# Encodings the DiT and SAME are published in. The conditioner, text encoder and tokenizer are
# always F32 -- they are small and quality-critical, so there is nothing to gain by quantizing them.
FLOAT_ENCODINGS = ("F16", "F32")
QUANT_ENCODINGS = ("Q4_K_M", "Q5_K_M", "Q8_0")
ENCODINGS = FLOAT_ENCODINGS + QUANT_ENCODINGS

# Training bases are published F16, F32 and Q4_K_M. Training on a quantized base works on every
# backend -- CPU, CUDA, Vulkan and Metal -- because the functional adapter path keeps the frozen base
# as a mul_mat argument, and ggml PR #4 taught out_prod(W, transpose(grad)) to take a quantized src0.
# It is also faster than F16 everywhere measured, on a 2.9x smaller file, and an adapter trained this
# way is audibly indistinguishable from an F16-trained control. See docs/TRAINING.md.
#
# Only Q4_K_M is published: it is the tier where the footprint win matters, and adding Q5_K_M/Q8_0
# bases would be three more large files for a saving nobody asked for. A request for a tier we do not
# publish falls back to F16 rather than failing, since "quantized inference plus a trainable base" is
# a reasonable ask. download_models.py --dry-run prints the resolved plan, so any substitution is
# visible rather than silent.
TRAINING_BASE_ENCODINGS = FLOAT_ENCODINGS + ("Q4_K_M",)
TRAINING_BASE_FALLBACK = "F16"


def dit_identity(variant, training_base=False):
    """Return the catalog identity embedded in a DiT GGUF."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown model variant: {variant}")
    model = f"stable-audio-3-{variant}"
    if training_base:
        model += "-base"
    return {
        "basename": f"{model}-dit",
        "name": f"{model} DiT",
        "finetune": f"{variant}-base" if training_base else variant,
        "upstream_repo": UPSTREAM_BASE_REPOS[variant] if training_base else None,
        "upstream_revision": UPSTREAM_BASE_REVISIONS[variant] if training_base else None,
    }


def dit_filename(variant, encoding, training_base=False):
    """Return the published DiT filename for a model family and encoding."""
    size, _ = VARIANTS[variant]
    identity = dit_identity(variant, training_base)
    return f"{identity['basename']}-{size}-{VERSION}-{encoding.upper()}.gguf"


def build_download_plan(namespace, variant, encoding, training_base=False):
    """Return ``[(repo_id, [filenames...]), ...]`` for download_models.py.

    Training needs the inference components as well as the base DiT, so
    ``training_base=True`` adds the base repository instead of replacing the
    normal inference repository.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown model variant: {variant}")
    enc = encoding.upper()
    if enc not in ENCODINGS:
        raise ValueError(
            f"unsupported encoding: {encoding} (expected one of {', '.join(e.lower() for e in ENCODINGS)})"
        )

    size, same = VARIANTS[variant]
    model = f"stable-audio-3-{variant}"
    plan = [
        (
            f"{namespace}/{model}-GGUF",
            [
                dit_filename(variant, enc),
                f"{model}-{same}-{VERSION}-{enc}.gguf",
                f"{model}-conditioner-{VERSION}-F32.gguf",
            ],
        )
    ]
    if training_base:
        base_enc = enc if enc in TRAINING_BASE_ENCODINGS else TRAINING_BASE_FALLBACK
        plan.append(
            (
                f"{namespace}/{model}-base-GGUF",
                [dit_filename(variant, base_enc, training_base=True)],
            )
        )
    plan.append(
        (
            f"{namespace}/{SHARED_REPO}",
            [
                f"t5gemma-b-b-ul2-encoder-0.3B-{VERSION}-F32.gguf",
                f"t5gemma-b-b-ul2-{VERSION}-vocab.gguf",
            ],
        )
    )
    return plan
