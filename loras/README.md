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

## bundled autoencoder adapters

Two SAME-S **decoder** adapters are checked in here, against the blanket
`*.safetensors` ignore — they are 1.36 MB each, where that rule exists for the
80 MB+ DiT exports sitting beside them, and having them in the branch means the
decoder-LoRA path can be exercised on any machine without fetching anything.

| file | base | rank | step |
|---|---|---|---|
| `same-s_declora_s4_step2000.safetensors` | SAME-S | 8 | 2000 |
| `same-s_declora_s4_step4000.safetensors` | SAME-S | 8 | 4000 |

They are **provisional** — two checkpoints from one run, kept while it is still
being decided whether they earn their place. SAME-S is not SAME-L: its artifact
sits in a different band and wants its own recipe, so do not read a SAME-L
result onto these.

No `.json` sidecar, and none is needed. `save_lora_safetensors` writes
`rank`/`alpha`/`adapter_type`/`target` into the file's own `__metadata__`, so
each converts on its own:

```bash
sa3-lora-convert --safetensors loras/same-s_declora_s4_step2000.safetensors \
                 --out models/lora-declora-s4-2000-f32.gguf
sa3-generate --model small-music --lora models/lora-declora-s4-2000-f32.gguf --prompt "..."
```

`--lora` needs no flag to place them: the `target: "decoder"` carried in the
checkpoint routes them onto the autoencoder. The published SAME-L counterpart
lives at [thepatch/same-l-decoder-lora](https://huggingface.co/thepatch/same-l-decoder-lora)
and is **not** interchangeable — loading it against SAME-S is refused on tensor
width rather than silently corrupting the model.
