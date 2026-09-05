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

SAOS_REPO = "stable-audio-open-small-GGUF"
SAOS_VARIANTS = ("arc", "kickbass", "jerry-grunge")
SAOS_ENCODINGS = ("F16", "Q8_0", "Q5_K_M", "Q4_K_M")
SAOS_DEFAULT_ENCODING = "Q5_K_M"

# Encodings the DiT and SAME are published in. The conditioner and tokenizer are always F32 --
# those two really are small (a 793 KiB conditioner, a 14 MiB vocab), so there is nothing to gain.
FLOAT_ENCODINGS = ("F16", "F32")
QUANT_ENCODINGS = ("Q4_K_M", "Q5_K_M", "Q8_0")
ENCODINGS = FLOAT_ENCODINGS + QUANT_ENCODINGS

# The text encoder is not small: at F32 it is 1074 MiB, the largest file in the set and bigger than
# a small-music DiT. Measured on small-music/Metal against an F32 control, swapping only the encoder:
#
#   F16   537 MiB  conditioning cos 1.000000000, generated-audio cos 0.999971  -- free
#   Q8_0  285 MiB  conditioning cos 0.999789,    generated-audio cos 0.991496  -- a real tradeoff
#   Q4_K_M 206 MiB conditioning cos 0.977908,    generated-audio cos 0.797692  -- not shipped
#
# Q4_K_M is excluded on evidence, not caution: it is 11.8% SLOWER than F32 for the encoder (the T5
# forward is too small to amortize dequant, unlike the DiT) while doing real damage to prompt
# fidelity, so Q8_0 dominates it outright. Note that training loss is nearly blind to all of this --
# Q4_K_M moved mean loss by 0.02% over 100 matched steps. Validate an encoder with conditioning
# cosine (SA3_DUMP_COND) and generated audio, never with a loss curve.
TEXT_ENCODER_ENCODINGS = ("F16", "F32", "Q8_0")
TEXT_ENCODER_DEFAULT = "F16"

# The autoencoder's default tier, on its own axis from the DiT's. F32 because SAME is the
# reference precision this project is checked against, because it is the last net the audio
# crosses (so whatever it adds reaches the output unmasked, and continuation/transform cross it
# twice per iteration), and because it is small enough beside the DiT for the bytes to be worth
# it. Quantized autoencoders are still published and still downloadable -- ask with --ae-encoding.
AUTOENCODER_DEFAULT = "F32"

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


def text_encoder_filename(encoding):
    """Return the published text-encoder filename for an encoding."""
    enc = encoding.upper()
    if enc not in TEXT_ENCODER_ENCODINGS:
        raise ValueError(
            f"unsupported text-encoder encoding: {encoding} "
            f"(expected one of {', '.join(e.lower() for e in TEXT_ENCODER_ENCODINGS)})"
        )
    return f"t5gemma-b-b-ul2-encoder-0.3B-{VERSION}-{enc}.gguf"


def build_download_plan(namespace, variant, encoding, training_base=False, text_encoding=None,
                        ae_encoding=None):
    """Return ``[(repo_id, [filenames...]), ...]`` for download_models.py.

    Training needs the inference components as well as the base DiT, so
    ``training_base=True`` adds the base repository instead of replacing the
    normal inference repository.

    ``encoding`` picks the DiT. The autoencoder has its own ``ae_encoding``
    (default F32) because it used to ride on ``encoding``, which meant asking
    for a quantized DiT also fetched a quantized SAME -- and then nothing else
    was on disk for the resolver to prefer. SAME is the last net the audio
    crosses and the cheap one to keep: 413 MB at F32 for SAME-S against 72 MB
    at Q4_K_M, beside a DiT that dominates either way.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown model variant: {variant}")
    enc = encoding.upper()
    if enc not in ENCODINGS:
        raise ValueError(
            f"unsupported encoding: {encoding} (expected one of {', '.join(e.lower() for e in ENCODINGS)})"
        )

    ae_enc = (ae_encoding or AUTOENCODER_DEFAULT).upper()
    if ae_enc not in ENCODINGS:
        raise ValueError(
            f"unsupported ae_encoding: {ae_encoding} (expected one of "
            f"{', '.join(e.lower() for e in ENCODINGS)})"
        )

    size, same = VARIANTS[variant]
    model = f"stable-audio-3-{variant}"
    plan = [
        (
            f"{namespace}/{model}-GGUF",
            [
                dit_filename(variant, enc),
                f"{model}-{same}-{VERSION}-{ae_enc}.gguf",
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
                text_encoder_filename(text_encoding or TEXT_ENCODER_DEFAULT),
                f"t5gemma-b-b-ul2-{VERSION}-vocab.gguf",
            ],
        )
    )
    return plan


def saos_dit_filename(variant, encoding):
    """Return the repo-relative published DiT path for SAOS or one of its finetunes."""
    variant = variant.lower().replace("_", "-")
    enc = encoding.upper()
    if variant not in SAOS_VARIANTS:
        raise ValueError(
            f"unknown SAOS variant: {variant} (expected one of {', '.join(SAOS_VARIANTS)})"
        )
    if enc not in SAOS_ENCODINGS:
        raise ValueError(
            f"unsupported SAOS encoding: {encoding} "
            f"(expected one of {', '.join(e.lower() for e in SAOS_ENCODINGS)})"
        )
    if variant == "arc":
        return f"stable-audio-open-small-dit-0.3B-{VERSION}-{enc}.gguf"
    if variant == "kickbass":
        return f"finetunes/kickbass/kickbass-v1-e257-dit-0.3B-{VERSION}-{enc}.gguf"
    return f"finetunes/jerry-grunge/jerry-grunge-bs64-step3000-dit-0.3B-{VERSION}-{enc}.gguf"


def saos_t5_filename(encoding):
    enc = encoding.upper()
    if enc not in SAOS_ENCODINGS:
        raise ValueError(f"unsupported SAOS T5 encoding: {encoding}")
    return f"t5-base-encoder-0.1B-{VERSION}-{enc}.gguf"


def saos_oobleck_filename(encoding):
    enc = encoding.upper()
    if enc not in SAOS_ENCODINGS:
        raise ValueError(f"unsupported SAOS Oobleck encoding: {encoding}")
    return f"stable-audio-open-small-oobleck-{VERSION}-{enc}.gguf"


def build_saos_download_plan(namespace, variant="arc", encoding=SAOS_DEFAULT_ENCODING,
                             text_encoding=None, ae_encoding=None):
    """Return the single-repository SAOS/finetune download plan."""
    enc = encoding.upper()
    text_enc = (text_encoding or enc).upper()
    ae_enc = (ae_encoding or enc).upper()
    return [(
        f"{namespace}/{SAOS_REPO}",
        [saos_dit_filename(variant, enc), saos_t5_filename(text_enc),
         saos_oobleck_filename(ae_enc)],
    )]
