# Module: QwenImagePipeline

## Purpose
The central orchestrator for image generation, editing, and decomposition. It manages the lifecycle of model components (Tokenizer, TextEncoder, UNet/Transformer, VAE, VisionTower) and executes the diffusion sampling loop.

## Key Files
- `Sources/QwenImage/Pipeline/QwenImagePipeline.swift`: Main pipeline logic.
- `Sources/QwenImage/Pipeline/QwenImageLayeredGeneration.swift`: Dedicated pipeline for layered image decomposition.
- `Sources/QwenImage/Pipeline/Scheduler.swift`: `QwenFlowMatchScheduler` implementation.
- `Sources/QwenImage/Pipeline/Parameters.swift`: Configuration structs.

## Architecture
- **Lazy Loading:** Heavily relies on `HubSnapshot` (from `swift-transformers`) to download and cache model weights.
- **Lifecycle Management:** Explicit methods (`releaseEncoders`, `releaseTokenizer`, `reloadTextEncoder`) allow external runtimes to manage memory pressure.
- **API Separation:** "Policy-Free" design separates the *Encoding* phase (`encodeGuidancePrompts`) from the *Generation* phase (`generatePixels`). This allows caching encodings.
- **Offloading:** Can unload "encoder-side" models after prompt encoding to free up memory for the UNet/Transformer.
- **Scheduling:** Implements a **Flow Matching** scheduler (`QwenFlowMatchScheduler`) with **Sigma Shifting**.
- **Quantization:** Supports both runtime quantization (on-the-fly) and loading pre-quantized weights.
- **LoRA:** Dynamic loading and application of LoRA adapters.

## Key Workflows
1.  **Text-to-Image:**
    - `generatePixels`:
        - Encodes prompt (TextEncoder) or accepts pre-computed `QwenGuidanceEncoding`.
        - Generates initial noise (Latents).
        - Loops `steps` times: `UNet.forward` -> `Scheduler.step`.
        - Decodes latents (VAE).
2.  **Layered Decomposition:**
    - `QwenLayeredPipeline.generate`:
        - Decomposes an image into foreground/background layers.
        - Uses "True Classifier-Free Guidance" (double forward pass).

## Dependencies
- **Internal:** `QwenImage/Model`, `QwenImage/Tokenizer`, `QwenImage/Weights`.
- **External:** `MLX` (Tensor ops), `CoreGraphics` (Image I/O).