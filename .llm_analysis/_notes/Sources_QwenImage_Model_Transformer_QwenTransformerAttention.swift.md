# Sources/QwenImage/Model/Transformer/QwenTransformerAttention.swift Analysis

## Purpose
- Joint Self-Attention mechanism.

## Key Observations
- **Concatenation**: Concatenates Text and Image Q/K/V tensors.
- **RoPE**: Applies 3D RoPE to the joint sequence.
- **Optimization**: Supports quantized attention (FlashAttention-like via MLXFast).

## Quality Assessment
- Critical performance path. Use of `MLXFast.scaledDotProductAttention` is good.
