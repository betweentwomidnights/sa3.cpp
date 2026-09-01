# loras

adapter files live here. two kinds:

- **converted `.gguf` adapters** - what `sa3-generate --lora <name>` loads (it resolves `lora-<name>-*.gguf`
  from the adapters dir, which defaults to the models dir; point it here with `SA3_ADAPTERS_DIR` /
  `--adapters-dir`). Runtime strength and multi-adapter blending are applied in weight space, not a static merge.
- **source exports** - a trained adapter's `.safetensors` + `.json` (or a raw `.ckpt`). This folder is the
  default `SA3_SOURCE_LORAS_DIR`.

## convert an export to a gguf (no python)

```bash
sa3-lora-convert --in loras/kev --out models/lora-kev-f32.gguf     # reads kev.safetensors + kev.json
```

`sa3-lora-convert` (and the `libsa3` `sa3_convert_lora` C ABI) are a C++ port of `tools/convert_lora.py`, so a
host can convert `.safetensors` adapters **in-process with no Python**. a raw `.ckpt` still needs the python /
pytorch helper `tools/lora_ckpt_export.py` to produce the `.safetensors`/`.json` pair first (a checkpoint is a
torch artifact).

adapter types dora-rows and bora are validated end-to-end against trained checkpoints (cossim 1.0); dora-cols and
the `-xs` variants are formula-validated. See the main README.

**a note about prompts.** sa3 loras really like it when you use prompts from your training data, so i use
helpers to populate 'dice' buttons in downstream apps. you don't have to take advantage of them, but i find it
super useful, especially when blending multiple loras (you'll get a prompt from any one of your loaded loras).
keep the `.txt` files you trained with in the same folder as your adapter and downstream apps (like the
[iplug2 demo](https://github.com/betweentwomidnights/sa3.cpp-iplug2-demo)) pick them up as a prompt pool; the
http server reads the `.json` pools in [`../prompts`](../prompts). (`libsa3` itself just generates — prompt
pools are the app's job.) 

## autoencoder adapters

An adapter whose checkpoint declares `target: "decoder"` (or `"encoder"`) is routed onto the
autoencoder instead of the DiT. `--lora` needs no extra flag to place it: `save_lora_safetensors`
writes `rank`/`alpha`/`adapter_type`/`target` into the file's own `__metadata__`, so an export
converts and loads on its own.

```bash
sa3-lora-convert --safetensors <export>.safetensors --out models/lora-<name>-f32.gguf
sa3-generate --model medium --lora models/lora-<name>-f32.gguf --prompt "..."
```

The SAME-L decoder adapter is published, with the write-up of what it does and how it was
trained, at [thepatch/same-l-decoder-lora](https://huggingface.co/thepatch/same-l-decoder-lora).

SAME-S and SAME-L adapters are **not** interchangeable — their artifacts sit in different bands
and each wants its own recipe. Loading one against the other is refused on tensor width rather
than silently corrupting the model, so a mismatch is an error and not a quiet quality loss.

A quantized autoencoder carrying an adapter merges through the host re-quantize path rather than
the graph path: the graph path reads `W` through `ggml_cast` and writes back through `ggml_cpy`,
and Metal has no f32 -> k-quant copy, so a Q4_K SAME-L used to abort with `unsupported op 'CPY'`.
See `lora_base_needs_host` in `src/lora.h`.
