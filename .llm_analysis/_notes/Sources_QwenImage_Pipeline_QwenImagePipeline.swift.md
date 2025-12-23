# Sources/QwenImage/Pipeline/QwenImagePipeline.swift Analysis

## Purpose
- The central orchestrator for Text-to-Image and Image Editing.

## Key Observations
- **Monolith**: Extremely large file (~2000 lines).
- **Responsibilities**:
    - Model Loading (TextEncoder, UNet, VAE, VisionTower).
    - Quantization setup.
    - LoRA loading (duplicated).
    - Encoding (Text + Vision injection).
    - Denoising Loops (`generatePixels`, `generateEditedPixels`).
- **Memory Management**: Explicit `offloadEncoderComponents` to save VRAM.
- **Vision Context**: Complex logic to inject vision tokens into the text prompt for editing.

## Recent Updates
- **API Separation**: Introduced `QwenGuidanceEncoding` and `generatePixels(..., guidanceEncoding: ...)` to separate the prompt encoding phase from the generation phase.
- **Optimization**: Added `progressRequested` flag to avoid costly `MLX.eval()` calls during the generation loop when no listener is attached.
- **Lifecycle Hooks**: Added explicit methods (`releaseEncoders`, `releaseTokenizer`, `reloadTextEncoder`) for fine-grained memory control by the Runtime layer.

## Quality Assessment
- **Refactoring Candidate**: Still a "God Class", but the API separation is a step towards breaking it down.
- **Performance**: The removal of implicit sync points makes it much more performant on GPU.