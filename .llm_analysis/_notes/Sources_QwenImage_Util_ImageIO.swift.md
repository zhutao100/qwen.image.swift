# Sources/QwenImage/Util/ImageIO.swift Analysis

## Purpose
- Handles Image Input/Output and Resizing.
- Bridges `CGImage` (CoreGraphics) and `MLXArray`.

## Key Observations
- **Custom Lanczos**: Implements a full Lanczos resampling algorithm in Swift (`resizeLanczosARGB`, `resizeLanczos`).
    - *Risk*: Performance might be an issue compared to optimized Metal/Accelerate kernels.
    - *Reasoning*: Likely done to ensure bit-exact matching with Python training pipelines or because MLX didn't have a matching resize kernel at the time.
- **Platform Specificity**: Heavily relies on `CoreGraphics` (`canImport(CoreGraphics)`).
- **Normalization**: helpers `normalizeForEncoder` ([-1, 1]) and `denormalizeFromDecoder`.

## Quality Assessment
- The custom Lanczos implementation is a significant chunk of code. If MLX offers a resize function, it should be considered to reduce maintenance burden, unless exact numerical matching is critical.
