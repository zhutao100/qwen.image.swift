# Sources/QwenImage/Model/Transformer/QwenTransformer.swift Analysis

## Purpose
- The core Diffusion Transformer (DiT) backbone.
- Implements the Multimodal Diffusion Transformer (MM-DiT) architecture.

## Key Observations
- **Joint Architecture**: Processes Text and Image tokens in parallel streams that interact via Joint Attention.
- **Conditioning**: Uses `QwenAdaLayerNormContinuous` for timestep conditioning.
- **RoPE**: Uses `QwenEmbedRope` for 3D positional embeddings (Time, Height, Width).

## Recent Updates
- **Performance**: Removed synchronization (`.item()`) in timestep embedding generation (`timeTextEmbed`). Now fully GPU-resident during the loop.

## Quality Assessment
- Clean implementation of the DiT architecture.
- Modularized into Blocks and Attention layers.