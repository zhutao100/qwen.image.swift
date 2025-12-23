# Sources/QwenImage/Pipeline/QwenImageLayeredGeneration.swift Analysis

## Purpose
- Dedicated pipeline for Layered Image Decomposition (`QwenLayeredPipeline`).

## Key Observations
- **Independence**: Operates largely independently of the main `QwenImagePipeline`.
- **Duplication**:
    - **Resize**: Implements `bilinearResizeLayered` (3rd resize implementation!).
    - **LoRA**: Implements `applyLora`, `loadLoraLayers` which appears nearly identical to the main pipeline's version.
    - **Loading**: Has its own `load` function re-implementing weight loading orchestration.
- **Logic**: Implements "True CFG" with CFG normalization.

## Recent Updates
- **API Improvements**: Added `LayeredPromptEncoding` struct and a `generate` overload that accepts pre-computed encodings. This supports the "Policy-Free" design, allowing runtimes to manage encoding caching.
- **Lifecycle Management**: Added explicit hooks (`releaseTextEncoder`, `releaseTokenizer`, `reloadTextEncoder`) to allow external control over memory usage.
- **Refactoring**: Explicitly split generation logic to support the new Runtime layer.

## Quality Assessment
- **High Duplication**: Significant code duplication with `QwenImagePipeline` persists.
- **Improved API**: The separation of encoding and generation is a solid architectural improvement.