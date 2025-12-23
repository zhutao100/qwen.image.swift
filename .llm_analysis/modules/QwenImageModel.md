# Module: QwenImageModel (Support Components)

## Sub-Module: TextEncoder (`Sources/QwenImage/Model/TextEncoder/`)
### Purpose
Processes text prompts and optional interleaved images to generate condition embeddings for the diffusion transformer. Based on the **Qwen** LLM architecture.

### Architecture
- **Backbone:** Standard Qwen Transformer (RMSNorm, SwiGLU, Rotary Embeddings).
- **Multimodal Capabilities:**
    - Can inject vision embeddings (from `VisionTower`) directly into the token sequence.
    - `replaceVisionTokens`: Replaces placeholder tokens (e.g., `<|image_pad|>`) with actual vision embeddings.
        - **Vectorized:** Recent updates use `MLX.where` and `MLX.take` for efficient, parallel token replacement.
- **Positional Embeddings:**
    - `buildPositionIds`: Generates **3D positional IDs** (Time, Height, Width) so the LLM understands the 2D/3D structure of inserted image patches.
    - `QwenRotaryEmbedding`: Applied to text and vision tokens alike.

## Sub-Module: VAE (`Sources/QwenImage/Model/VAE/`)
### Purpose
Variational Autoencoder for compressing images into a lower-dimensional latent space to enable efficient diffusion.

### Architecture
- **3D Causal CNN:** Uses 3D convolutions (`QwenImageEncoder3D`, `QwenImageDecoder3D`), likely inherited from a video generation architecture (Video VAE).
- **Latent Space:**
    - Encodes RGB images `[B, 3, H, W]` to Latents `[B, 16, H/8, W/8]`.
    - **Normalization:** Applies static mean/std normalization to latents (`latentsMean`, `latentsStd`).
- **Quantization Compatibility:** Includes `quantConv` and `postQuantConv` layers (channel projection).

## Sub-Module: Vision (`Sources/QwenImage/Model/TextEncoder/Vision/`)
- **QwenVisionTower:** Encodes reference images into patch embeddings that are fed into the TextEncoder or used for cross-attention.
- **Components:** `QwenVisionPatchEmbed`, `QwenVisionAttention`, `QwenVisionMLP`.
- **Optimization:** Window attention mask generation and index calculations are now fully vectorized in MLX.