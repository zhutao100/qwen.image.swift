Below is what the **on-repo file layout** looks like for the three major Qwen-Image variants on Hugging Face, and how each piece maps onto a **Diffusers pipeline + Transformers text (and optionally vision) stack**.

---

## 1) Common packaging convention (Diffusers-style)

All three models are published as **Diffusers pipelines** with a top-level `model_index.json` that declares:

* which pipeline class to instantiate (`_class_name`)
* which submodules exist (scheduler / transformer / vae / text encoder / tokenizer / processor)
* which library each submodule comes from (`diffusers` vs `transformers`)

You can see this directly in:

* **Qwen-Image** `model_index.json` → `_class_name: "QwenImagePipeline"` ([Hugging Face][1])
* **Qwen-Image-Edit-2511** `model_index.json` → `_class_name: "QwenImageEditPlusPipeline"` ([Hugging Face][2])
* **Qwen-Image-Layered** `model_index.json` → `_class_name: "QwenImageLayeredPipeline"` ([Hugging Face][3])

---

## 2) Qwen/Qwen-Image (text-to-image) — repository tree

Top-level structure (abridged): ([Hugging Face][4])

```text
Qwen-Image/
  model_index.json
  README.md
  LICENSE
  .gitattributes
  scheduler/
    scheduler_config.json
  text_encoder/
    config.json
    generation_config.json
    model-00001-of-00004.safetensors
    ...
    model.safetensors.index.json
  tokenizer/
    merges.txt
    vocab.json
    tokenizer_config.json
    ...
  transformer/
    config.json
    diffusion_pytorch_model-00001-of-00009.safetensors
    ...
    diffusion_pytorch_model.safetensors.index.json
  vae/
    config.json
    diffusion_pytorch_model.safetensors
```

### What each component does (Qwen-Image)

* **`model_index.json`**: the *manifest* that Diffusers uses to load the right pipeline class and wire submodules. It explicitly binds:

  * `scheduler = FlowMatchEulerDiscreteScheduler` (Diffusers)
  * `text_encoder = Qwen2_5_VLForConditionalGeneration` (Transformers)
  * `tokenizer = Qwen2Tokenizer` (Transformers)
  * `transformer = QwenImageTransformer2DModel` (Diffusers)
  * `vae = AutoencoderKLQwenImage` (Diffusers) ([Hugging Face][1])

* **`scheduler/`**: sampling schedule + sigma/timestep logic. Here it’s a **flow-matching Euler discrete scheduler** with dynamic shifting parameters (e.g., `max_image_seq_len`, `time_shift_type`, etc.). ([Hugging Face][5])

* **`transformer/`**: the **main diffusion transformer** weights and architecture config. The config shows it’s a `QwenImageTransformer2DModel` with (notably) `num_layers: 60`, `num_attention_heads: 24`, RoPE axis dims, etc. ([Hugging Face][6])

* **`vae/`**: latent autoencoder (`AutoencoderKLQwenImage`) used to decode final latents into pixels (and encode if needed). Config exposes latent stats (`latents_mean/std`) and latent dimensionality (e.g., `z_dim`). ([Hugging Face][7])

* **`text_encoder/` + `tokenizer/`**: a Qwen2.5-VL family text backbone (`Qwen2_5_VLForConditionalGeneration`) plus its tokenizer. In Diffusers usage, this typically runs in an “embed prompts / conditioning” role rather than full autoregressive generation, but it’s packaged as a standard Transformers model (hence `generation_config.json` and sharded `model-0000x-of-00004.safetensors`). ([Hugging Face][1])

* **Sharded weights + `*.index.json`**: `diffusion_pytorch_model-000xx-of-000yy.safetensors` (and the corresponding `*.index.json`) are the standard HF sharding format; the index maps parameter names → shard files so loaders can stream/dispatch efficiently. ([Hugging Face][8])

---

## 3) Qwen/Qwen-Image-Edit-2511 (image editing) — repository tree

Top-level structure (abridged): ([Hugging Face][9])

```text
Qwen-Image-Edit-2511/
  model_index.json
  README.md
  .gitattributes
  processor/
    preprocessor_config.json
    video_preprocessor_config.json
    chat_template.jinja
    tokenizer.json
    merges.txt
    vocab.json
    ...
  scheduler/
    scheduler_config.json
  text_encoder/           (Transformers)
    config.json
    generation_config.json
    model-00001-of-00004.safetensors
    ...
    model.safetensors.index.json
  tokenizer/              (Transformers)
    merges.txt
    vocab.json
    tokenizer_config.json
    ...
  transformer/            (Diffusers)
    config.json
    diffusion_pytorch_model-00001-of-00005.safetensors
    ...
    diffusion_pytorch_model.safetensors.index.json
  vae/
    config.json
    diffusion_pytorch_model.safetensors
```

### What’s “new” vs Qwen-Image (and why)

* **`processor/` (Qwen2VLProcessor)** appears in the manifest for Edit: ([Hugging Face][2])
  This is the key packaging difference for edit pipelines: the processor bundles **tokenization + vision pre-processing + chat templating** in one place, so the pipeline can accept **image inputs** (and possibly video-style preproc knobs) in addition to text. You can see `preprocessor_config.json`, `video_preprocessor_config.json`, and `chat_template.jinja` in the folder. ([Hugging Face][10])

* The pipeline class changes to **`QwenImageEditPlusPipeline`**, which is what `DiffusionPipeline.from_pretrained(...)` will instantiate when it reads `model_index.json`. ([Hugging Face][2])

Everything else (scheduler / text encoder / transformer / VAE) is conceptually the same set of building blocks, but tuned and wired for **image-to-image editing** rather than pure text-to-image.

---

## 4) Qwen/Qwen-Image-Layered (Dec 2025, “layered” generation) — repository tree

Top-level structure (abridged): ([Hugging Face][11])

```text
Qwen-Image-Layered/
  model_index.json
  README.md
  .gitattributes
  processor/              (Transformers: Qwen2VLProcessor)
    preprocessor_config.json
    chat_template.jinja
    tokenizer.json
    ...
  scheduler/
    scheduler_config.json
  text_encoder/
    config.json
    generation_config.json
    model-00001-of-00004.safetensors
    ...
    model.safetensors.index.json
  tokenizer/
    merges.txt
    vocab.json
    tokenizer_config.json
    ...
  transformer/
    config.json
    diffusion_pytorch_model-00001-of-00005.safetensors
    ...
    diffusion_pytorch_model.safetensors.index.json
  vae/
    config.json
    diffusion_pytorch_model.safetensors
```

### What’s distinctive in the Layered variant

* **Manifest/pipeline**: `model_index.json` points to **`QwenImageLayeredPipeline`** and includes a **`processor`** (again `Qwen2VLProcessor`). ([Hugging Face][3])

* **Transformer config flags hint at the “layered” mechanism**:

  * `use_layer3d_rope: true`
  * `use_additional_t_cond: true` ([Hugging Face][12])

  Even without reading the pipeline implementation, these are strong indicators that the diffusion transformer is encoding something beyond plain 2D spatial structure—e.g., an additional “layer” axis handled via 3D RoPE or similar positional scheme.

* **Scheduler config differs slightly from base Qwen-Image**:

  * Layered scheduler uses `time_shift_type: "linear"` and `shift_terminal: false` ([Hugging Face][13])
  * Base Qwen-Image shows `time_shift_type: "exponential"` and a numeric `shift_terminal` ([Hugging Face][5])

* **VAE class and structure remain the same family** (`AutoencoderKLQwenImage`) and include the latent normalization stats in config. ([Hugging Face][14])

---

## 5) Practical walkthrough: how loading works end-to-end

At runtime, **Diffusers** roughly does:

1. Download repo snapshot
2. Read `model_index.json` to determine:

   * pipeline class (`QwenImagePipeline` / `QwenImageEditPlusPipeline` / `QwenImageLayeredPipeline`)
   * which subfolders to load and from which library ([Hugging Face][1])
3. Instantiate:

   * Scheduler from `scheduler/scheduler_config.json` ([Hugging Face][5])
   * Diffusion transformer from `transformer/config.json` + sharded safetensors ([Hugging Face][6])
   * VAE from `vae/config.json` + safetensors ([Hugging Face][7])
   * Text encoder/tokenizer (and optionally processor) via Transformers configs/weights ([Hugging Face][3])

---

## 6) The Hugging Face endpoints you referenced (verbatim)

```text
https://huggingface.co/Qwen/Qwen-Image-Layered
https://huggingface.co/api/models/Qwen/Qwen-Image-Layered
https://huggingface.co/Qwen/Qwen-Image-Layered/raw/main/model_index.json
```

---

If you want, I can go one level deeper and **diff the three `transformer/config.json` files field-by-field** (what changed between Qwen-Image vs Edit-2511 vs Layered), then map those changes to likely architectural intent (conditioning channels, extra temporal/layer conditioning, RoPE axes semantics, etc.).

[1]: https://huggingface.co/Qwen/Qwen-Image/blob/main/model_index.json "model_index.json · Qwen/Qwen-Image at main"
[2]: https://huggingface.co/Qwen/Qwen-Image-Edit-2511/blob/main/model_index.json "model_index.json · Qwen/Qwen-Image-Edit-2511 at main"
[3]: https://huggingface.co/Qwen/Qwen-Image-Layered/blob/main/model_index.json "model_index.json · Qwen/Qwen-Image-Layered at main"
[4]: https://huggingface.co/Qwen/Qwen-Image/tree/main "Qwen/Qwen-Image at main"
[5]: https://huggingface.co/Qwen/Qwen-Image/blob/main/scheduler/scheduler_config.json "scheduler/scheduler_config.json · Qwen/Qwen-Image at main"
[6]: https://huggingface.co/Qwen/Qwen-Image/blob/main/transformer/config.json "transformer/config.json · Qwen/Qwen-Image at main"
[7]: https://huggingface.co/Qwen/Qwen-Image/blob/main/vae/config.json "vae/config.json · Qwen/Qwen-Image at main"
[8]: https://huggingface.co/Qwen/Qwen-Image/tree/main/transformer "Qwen/Qwen-Image at main"
[9]: https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/main "Qwen/Qwen-Image-Edit-2511 at main"
[10]: https://huggingface.co/Qwen/Qwen-Image-Edit-2511/tree/main/processor "Qwen/Qwen-Image-Edit-2511 at main"
[11]: https://huggingface.co/Qwen/Qwen-Image-Layered/tree/main "Qwen/Qwen-Image-Layered at main"
[12]: https://huggingface.co/Qwen/Qwen-Image-Layered/blob/main/transformer/config.json "transformer/config.json · Qwen/Qwen-Image-Layered at main"
[13]: https://huggingface.co/Qwen/Qwen-Image-Layered/blob/main/scheduler/scheduler_config.json "scheduler/scheduler_config.json · Qwen/Qwen-Image-Layered at main"
[14]: https://huggingface.co/Qwen/Qwen-Image-Layered/blob/main/vae/config.json "vae/config.json · Qwen/Qwen-Image-Layered at main"
