# Sources/QwenImage/Util/LatentUtilities.swift Analysis

## Purpose
- Utilities for manipulating latent representations.

## Key Functions
- `packLatents`: Reshapes [B, C, H, W] -> [B, H*W/256, C*4] (Spatial to Channel packing).
- `unpackLatents`: Inverse of pack.

## Quality Assessment
- Uses `precondition` for runtime safety checks on tensor shapes.
- Clear and concise.
