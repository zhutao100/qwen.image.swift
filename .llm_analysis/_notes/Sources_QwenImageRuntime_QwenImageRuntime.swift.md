# Sources/QwenImageRuntime/QwenImageRuntime.swift Analysis

## Purpose
- **New Middleware Module**: Defines the "Policy Layer" of the architecture.
- Decouples the "mechanism" (in `QwenImage`) from the "policy" (memory limits, caching strategy).

## Key Components
- **Sessions**: `ImagePipelineSession` and `LayeredPipelineSession` wrap core pipelines. They enforce policies like "Release Text Encoder after encoding" to save VRAM.
- **Caching**:
    - `PromptEmbeddingsCache`: Caches text encoder outputs based on model ID, prompt, and quantization.
    - `LayeredCaptionCache`: Caches caption embeddings for layered generation, using **image hashing** to key conditioning on specific input images without re-processing.
- **GPU Policy**: `GPUCachePolicy` manages MLX memory limits with presets (e.g., "Low Memory", "High Memory").

## Quality Assessment
- **Excellent Architecture**: This module solves the "God Class" problem in the core library by moving state management and policy decisions out to a dedicated layer.
- **Thread Safety**: Uses Swift Actors (`actor`) for session and cache management, ensuring thread safety in concurrent environments (like the App).
