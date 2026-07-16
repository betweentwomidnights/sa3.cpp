# LoRA Checkpoint Contract Audit

Source-of-truth files: `src/lora.h`, `src/lora_convert.h`, `src/train_checkpoint.h`,
`src/train_resume.h`, `tools/convert_lora.py`, `tools/lora_ckpt_export.py`, and
`tools/lora_spec_test.py`.

## GGUF metadata

A training checkpoint must be a GGUF adapter with:

- `general.architecture = "sa3-lora"`
- `general.name = "sa3 <adapter_type> adapter"` or equivalent descriptive name
- `lora.adapter_type`: one of the supported adapter families below
- `lora.rank`: nonzero integer
- `lora.alpha`: float
- `lora.n_targets`: count of distinct mapped DiT target modules

`src/lora.h::load_lora()` requires `lora.rank` and `lora.alpha`; `lora.adapter_type` defaults to `lora` if absent, but new checkpoints should always write it.

## Trainer-state sidecars

Native step checkpoints are immutable pairs: the inference-ready `adapter-step-N.gguf` described
above and a host-only `trainer-state-step-N.gguf`. The sidecar is never passed to the inference
loader. It uses `general.architecture = "sa3-trainer-state"` and records:

- `trainer.state_version`, optimizer step, epoch, next-sample cursor, and shuffled dataset order;
- the ordered target inventory and AdamW `m`/`v` tensors for every trainable adapter component;
- serialized shuffle, crop, CFG-dropout, prompt, inpainting, and diffusion RNG/distribution state;
- a trajectory-compatibility fingerprint covering the model, dataset, targets, and math-affecting
  training configuration;
- the paired adapter filename and its file fingerprint, preventing moments from being combined with
  a different adapter that merely has compatible tensor shapes.

The adapter temporary is published first and the state sidecar last, making the sidecar the marker
for a complete resumable pair. `adapter-final.gguf` remains an inference convenience copy; the final
optimizer update is also emitted as a numbered resumable pair.

## Tensor names and shapes

For each adapted base DiT weight named `<stem>.weight`, adapter tensors are named `<stem>.<kind>`.

Standard low-rank families:

- `<stem>.lora_A`: shape `[rank, in]` in PyTorch/safetensors order; loaded by ggml as `ne=[in, rank]`.
- `<stem>.lora_B`: shape `[out, rank]` in PyTorch/safetensors order; loaded by ggml as `ne=[rank, out]`.
- `<stem>.magnitude`: required for `dora-rows` (`[out]`) and `dora-cols` (`[in]`).
- `<stem>.magnitude_r`: required for `bora` (`[out]`).
- `<stem>.magnitude_c`: required for `bora` (`[in]`).

XS families:

- `<stem>.U`: shape `[out, rank]`.
- `<stem>.V`: shape `[in, rank]`.
- `<stem>.M_xs`: shape `[rank, rank]`.
- Plus the same magnitude tensors required by the corresponding normalized family.

Runtime formulas are pinned by `tools/lora_spec_test.py`: `lora`/`lora-xs` are additive; `dora-rows` normalizes rows and uses magnitude `[out]`; `dora-cols` normalizes columns and uses magnitude `[in]`; `bora` applies row normalization with `magnitude_r` then column normalization with `magnitude_c`.

## Supported adapter families

The loader contract supports:

- `lora`
- `dora-rows`
- `dora-cols`
- `bora`
- feasible `-xs` variants: `lora-xs`, `dora-rows-xs`, `dora-cols-xs`, `bora-xs`

Training initialization rejects ranks larger than either dimension of a target weight. The ggml training graph implements every family above, including the `-xs` variants: for `-xs`, `U`/`V` are frozen top-rank SVD bases of the base weight and only the `M_xs` core (plus any DoRA/BoRA magnitudes) is trained, matching the reference forward `delta = U @ M_xs @ V^T`. By default `sa3-train` computes the SVD bases natively (randomized SVD); `--svd-bases <bases.gguf>` loads precomputed bases instead (generate them with `tools/compute_svd_bases.py` for exact `torch.linalg.svd` parity). Checkpoint loading/generation retains the broader host-side adapter contract above.

## DiT target naming rules

Training checkpoints must use existing DiT GGUF stems, not PyTorch module names. Export/conversion maps a PyTorch LoRA key:

`<module>.parametrizations.weight.0.<kind>`

where `<kind>` is one of `lora_A`, `lora_B`, `magnitude`, `magnitude_r`, `magnitude_c`, `U`, `V`, `M_xs`, by:

1. accepting only modules beginning with `model.`;
2. constructing `model.model.<module_without_model_prefix>.weight`;
3. applying the DiT rename table from `src/lora_convert.h::dit_rename()` / `tools/convert_dit.py`;
4. stripping the final `.weight` to get `<stem>`;
5. writing `<stem>.<kind>`.

Unmapped or non-DiT modules, such as conditioners, are skipped.

Mapped DiT stems include top-level weights such as `dit.pre_conv`, `dit.post_conv`, `dit.cond_embed.{0,2}`, `dit.global_embed.{0,2}`, `dit.proj_in`, `dit.proj_out`, and per-layer stems such as `dit.<i>.self.qkv`, `dit.<i>.self.out`, `dit.<i>.cross.q`, `dit.<i>.cross.kv`, `dit.<i>.cross.out`, `dit.<i>.ff.proj`, `dit.<i>.ff.out`, `dit.<i>.local.{0,2}` when their source names map to `.weight` tensors. Biases/gammas/memory tokens are present in the rename table but are not LoRA weight targets unless they map from a parametrized `.weight` module.

## Loader filename contract

`sa3-generate --lora <path>` accepts an existing file directly. `--lora <name>` resolves in the adapters directory as `lora-<name>-*.gguf`. Training outputs intended for bare-name loading should therefore be named like `lora-<run-name>-stepNNNN.gguf` or `lora-<run-name>-final.gguf`. Any `.gguf` path remains valid for explicit path loading.
