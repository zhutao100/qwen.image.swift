# Sources/QwenImage/Model/Transformer/QwenTransformerBlock.swift Analysis

## Purpose
- A single block of the MM-DiT.

## Key Observations
- **Modulation**: AdaLN-Zero style modulation (Scale, Shift, Gate) applied to normalization layers.
- **Dual Streams**: Maintains separate streams for Image and Text, fusing them only in Attention.

## Quality Assessment
- Follows standard MM-DiT design patterns.
