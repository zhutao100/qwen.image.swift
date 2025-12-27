## 1) Qwen-Image family at a glance

The **Qwen-Image** family is Qwen’s diffusion-based image generation/editing stack, emphasizing **complex text rendering** (notably strong for Chinese) and **instruction-following edits**. ([Hugging Face][1])

The three variants you called out map cleanly onto three “modes” of use in Diffusers:

| Model                       | Primary task                                                             | Diffusers pipeline class                        |
| --------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------- |
| `Qwen/Qwen-Image`           | text-to-image generation                                                 | `QwenImagePipeline` ([Hugging Face][2])         |
| `Qwen/Qwen-Image-Edit-2511` | **multi-image** instruction editing + better consistency                 | `QwenImageEditPlusPipeline` ([Hugging Face][3]) |
| `Qwen/Qwen-Image-Layered`   | **layer decomposition** into multiple RGBA layers (inherent editability) | `QwenImageLayeredPipeline` ([Hugging Face][4])  |

---

## 2) Architecture: what’s actually inside these checkpoints

### 2.1 Core components (shared pattern across the family)

All three are packaged in **Diffusers “multi-folder” format** (so you’ll see `transformer/`, `vae/`, `text_encoder/`, etc.). For example, `Qwen/Qwen-Image` ships the standard set: `scheduler/`, `text_encoder/`, `tokenizer/`, `transformer/`, `vae/`, plus `model_index.json`. ([Hugging Face][5])

At a high level:

1. **Tokenizer + text encoder (Qwen2.5-VL)**
   The `model_index.json` for Qwen-Image points `text_encoder` to `Qwen2_5_VLForConditionalGeneration`, with `Qwen2Tokenizer` for text. ([Hugging Face][2])

2. **Diffusion “transformer” backbone (DiT-style, dual-stream blocks)**
   Diffusers exposes this as `QwenImageTransformer2DModel`, described as using **dual stream DiT blocks** (60 layers in the reference config), with 24 heads and head dim 128. ([Hugging Face][6])

3. **VAE for latent ↔ pixel space**
   Qwen-Image uses `AutoencoderKLQwenImage` as its VAE component. ([Hugging Face][2])

4. **Flow-matching scheduler**
   The pipelines are configured with `FlowMatchEulerDiscreteScheduler`. ([Hugging Face][2])

### 2.2 What changes for Edit / Layered variants

#### Qwen-Image-Edit (and Edit-2511): “dual-path” image conditioning

Qwen-Image-Edit is explicitly described as feeding the input image into **Qwen2.5-VL for semantic control** and **the VAE encoder for appearance control**, so the model can handle both semantic edits (“turn the cat into a dog”) and appearance-preserving edits (“keep identity, change outfit”). ([Hugging Face][7])

`Qwen-Image-Edit-2511` is a later “Edit Plus” style model and is invoked via `QwenImageEditPlusPipeline` (multi-image list input in the model card’s quickstart). ([Hugging Face][8])

#### Qwen-Image-Layered: decomposition into multiple RGBA layers

Qwen-Image-Layered decomposes a single input image into **N RGBA layers**, enabling edits that only affect one layer while keeping others stable (“physical isolation” for consistency). It supports **variable layer counts** and **recursive decomposition** (decompose a layer again). ([Hugging Face][9])

---

## 3) “Walkthrough” in practice (Diffusers)

### 3.1 Install / versioning (important)

The official model cards and Diffusers docs repeatedly assume **Diffusers main/dev** (installed from GitHub) rather than a stable PyPI-only workflow. ([Hugging Face][1])

```bash
pip install git+https://github.com/huggingface/diffusers
# For Layered export helpers (as shown in the Layered quickstart):
pip install python-pptx
```

Layered also calls out `transformers>=4.51.3` (to support Qwen2.5-VL). ([Hugging Face][9])

---

## 4) Variant-by-variant technical walkthrough

### 4.1 Qwen/Qwen-Image (text-to-image)

**Key knobs you’ll see in official examples**

* Use an “empty” negative prompt (`" "`) to enable CFG computations.
* Use `true_cfg_scale` (not `guidance_scale`) for classic classifier-free guidance. ([Hugging Face][1])
* The model card provides recommended “bucketed” aspect ratio sizes (e.g., 1328×1328, 1664×928, etc.). ([Hugging Face][1])

Minimal code (matches the model card pattern):

```python
from diffusers import DiffusionPipeline
import torch

model_id = "Qwen/Qwen-Image"

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)

prompt = 'A coffee shop entrance with a chalkboard sign reading "Qwen Coffee $2 per cup"'
negative_prompt = " "  # enables CFG computations per docs

width, height = 1664, 928  # 16:9 bucket from the model card
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

image.save("qwen_image_t2i.png")
```

(These parameter choices mirror the official quickstart.) ([Hugging Face][1])

---

### 4.2 Qwen/Qwen-Image-Edit-2511 (multi-image edit, improved consistency)

**What it’s for**

* It is described as an improved successor to Edit-2509 with “notably better consistency,” including mitigating drift and improving character consistency; it also mentions integrated LoRA capabilities and strengthened geometric reasoning. ([Hugging Face][8])

**How you call it**

* Use `QwenImageEditPlusPipeline`.
* Pass `image=[image1, image2, ...]` (list input) as shown in the official quickstart. ([Hugging Face][8])

```python
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    torch_dtype=torch.bfloat16,
).to("cuda")

image1 = Image.open("input1.png")
image2 = Image.open("input2.png")

out = pipe(
    image=[image1, image2],
    prompt="The magician bear is on the left, the alchemist bear is on the right, facing each other.",
    generator=torch.manual_seed(0),
    true_cfg_scale=4.0,
    negative_prompt=" ",
    num_inference_steps=40,
    guidance_scale=1.0,  # present in the official example, but see note below
).images[0]

out.save("qwen_image_edit_2511.png")
```

This is essentially the model card’s snippet. ([Hugging Face][8])

**Important guidance note (Diffusers): `guidance_scale` vs `true_cfg_scale`**
Diffusers explicitly notes that `guidance_scale` is currently a placeholder for *future guidance-distilled* variants and is ineffective for the standard pipelines; to enable CFG you should pass `true_cfg_scale > 1` and provide `negative_prompt` (even `" "`). ([Hugging Face][10])

---

### 4.3 Qwen/Qwen-Image-Layered (RGBA layer decomposition)

**What it does**

* Decomposes an input image into multiple RGBA layers to enable “inherent editability,” and supports operations like resizing/repositioning/recoloring per-layer without disturbing the rest. ([Hugging Face][9])
* Supports variable layer counts and recursive decomposition. ([Hugging Face][9])

**How you call it**

* Use `QwenImageLayeredPipeline`.
* Provide an RGBA image, set `layers`, and pick a `resolution` bucket (the model card recommends 640 for this version). ([Hugging Face][9])

```python
import torch
from PIL import Image
from diffusers import QwenImageLayeredPipeline

pipe = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered").to("cuda", torch.bfloat16)

rgba = Image.open("input.png").convert("RGBA")

out = pipe(
    image=rgba,
    layers=4,
    resolution=640,
    true_cfg_scale=4.0,
    negative_prompt=" ",
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(777),
    cfg_normalize=True,
    use_en_prompt=True,
).images[0]

# The model card shows saving each returned layer image
for i, layer_img in enumerate(out):
    layer_img.save(f"layer_{i}.png")
```

The arguments shown (`layers`, `resolution`, `cfg_normalize`, `use_en_prompt`) come directly from the official quickstart. ([Hugging Face][9])

---

## 5) Practical engineering notes

### 5.1 File layout and what each folder is for

From the published trees/model_index entries:

* `transformer/`: the diffusion transformer backbone (`QwenImageTransformer2DModel`). ([Hugging Face][5])
* `vae/`: VAE (`AutoencoderKLQwenImage`) for latent compression and decode. ([Hugging Face][2])
* `text_encoder/` + `tokenizer/`: Qwen2.5-VL-based text conditioning. ([Hugging Face][2])
* `processor/` appears in Edit-2511 and Layered and corresponds to Qwen2.5-VL style preprocessing for image+text inputs. ([Hugging Face][3])
* `scheduler/`: flow-matching scheduler config. ([Hugging Face][2])

### 5.2 CFG control: always decide deliberately

If you take nothing else away from the Diffusers docs: for current Qwen-Image pipelines, **use `true_cfg_scale` + `negative_prompt`** to enable CFG; do not expect `guidance_scale` to do anything unless you are on a guidance-distilled variant. ([Hugging Face][10])

### 5.3 Resource sizing and acceleration options

In the ComfyUI-native workflow documentation, the fp8 variant of Qwen-Image is shown running on a 24GB-class GPU (RTX 4090D) with a “VRAM usage reference” table, and it also lists the storage sizes for bf16 vs fp8 checkpoints. ([ComfyUI][11])
Separately, Diffusers docs highlight an official “Lightning” LoRA approach to reduce steps (example: 8 steps) for faster inference. ([Hugging Face][10])

---

If you tell me your target runtime (Diffusers-only vs ComfyUI vs a production server like vLLM/SGLang) and the GPU class you’re deploying on, I can recommend a concrete config profile (dtype, steps, CFG settings, batching, and whether a Lightning/distilled path is worth it) tailored to that environment.

[1]: https://huggingface.co/Qwen/Qwen-Image "Qwen/Qwen-Image · Hugging Face"
[2]: https://huggingface.co/Qwen/Qwen-Image/blob/main/model_index.json "model_index.json · Qwen/Qwen-Image at main"
[3]: https://huggingface.co/Qwen/Qwen-Image-Edit-2511/blob/main/model_index.json "model_index.json · Qwen/Qwen-Image-Edit-2511 at main"
[4]: https://huggingface.co/Qwen/Qwen-Image-Layered/blob/main/model_index.json "model_index.json · Qwen/Qwen-Image-Layered at main"
[5]: https://huggingface.co/Qwen/Qwen-Image/tree/main "Qwen/Qwen-Image at main"
[6]: https://huggingface.co/docs/diffusers/main/api/models/qwenimage_transformer2d "QwenImageTransformer2DModel"
[7]: https://huggingface.co/Qwen/Qwen-Image-Edit "Qwen/Qwen-Image-Edit · Hugging Face"
[8]: https://huggingface.co/Qwen/Qwen-Image-Edit-2511 "Qwen/Qwen-Image-Edit-2511 · Hugging Face"
[9]: https://huggingface.co/Qwen/Qwen-Image-Layered "Qwen/Qwen-Image-Layered · Hugging Face"
[10]: https://huggingface.co/docs/diffusers/main/en/api/pipelines/qwenimage "QwenImage"
[11]: https://docs.comfy.org/tutorials/image/qwen/qwen-image "Qwen-Image ComfyUI Native Workflow Example - ComfyUI"
