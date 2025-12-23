# Module: QwenImageTransformer

## Purpose
The core diffusion model backbone. It implements a Joint Transformer (MM-DiT style) where text and image tokens are processed together in a single sequence, allowing for dense multimodal interaction.

## Key Files
- `QwenTransformer.swift`: Main transformer model definition.
- `UNet.swift`: Wrapper handling latent packing/unpacking and providing the standard "UNet-like" interface for the pipeline.
- `QwenTransformerBlock.swift`: Single transformer layer with adaptive modulation (AdaLN-Continuous).
- `QwenTransformerAttention.swift`: Joint self-attention implementation.
- `QwenEmbedRope.swift`: 3D Rotary Positional Embeddings (Time, Height, Width).

## Architecture
- **Joint Architecture:** Text and image tokens are concatenated into a single sequence `[Text Tokens, Image Tokens]`.
- **Modulation:** Uses **AdaLN-Continuous** (Adaptive LayerNorm).
- **Positional Embeddings:**
    - **Rotary (RoPE):** Applied to queries and keys. Specialized "3D" RoPE handles the spatial structure of images (H, W) alongside text positions.
- **Performance:** Synchronization points (`MLX.eval()`, `.item()`) have been removed from the hot loops (like timestep embedding generation) to ensure full GPU residency during inference.
- **Quantization Support:** Explicit hooks (`setAttentionQuantization`, `computeQuantizedAttention`) for quantized attention computation (KV cache quantization).

## Dependencies
- `MLX`, `MLXNN`.
- `AttentionUtils` (shared utility for attention mechanisms).