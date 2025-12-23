# Module: QwenImageRuntime

## Purpose
The **QwenImageRuntime** module serves as the "Policy Layer" of the architecture. It acts as middleware between the core library (`QwenImage`) and consumer applications (`QwenImageApp`, `QwenImageCLI`). Its primary goal is to manage state, caching, and resource lifecycle policies that the core library should not be opinionated about.

## Key Files
- `Sources/QwenImageRuntime/QwenImageRuntime.swift`: Module entry point.
- `Sources/QwenImageRuntime/ImagePipelineSession.swift`: Actor-based session for Text-to-Image/Editing.
- `Sources/QwenImageRuntime/LayeredPipelineSession.swift`: Actor-based session for Layered Generation.
- `Sources/QwenImageRuntime/PromptEmbeddingsCache.swift`: Caching for text encodings.
- `Sources/QwenImageRuntime/LayeredCaptionCache.swift`: Caching for layered conditioning.
- `Sources/QwenImageRuntime/GPUCachePolicy.swift`: MLX memory management utilities.

## Architecture
- **Policy vs. Mechanism:**
    - `QwenImage` (Core) provides the *mechanism* (e.g., "encode this prompt", "generate pixels").
    - `QwenImageRuntime` provides the *policy* (e.g., "cache this encoding", "unload encoder to save RAM").
- **Session-Based:** Consumers interact with **Sessions** (Actors) rather than raw pipelines. Sessions are thread-safe and handle state management.
- **Caching:**
    - **Prompt Embeddings:** Caches the output of the Text Encoder based on Model ID, Revision, Quantization, and Prompt text.
    - **Layered Captions:** Caches conditioning for layered generation. Uniquely uses **image hashing** (from original file bytes) to key the cache, allowing re-use of encodings for the same input image without re-running the VAE/VisionTower.
- **Resource Management:**
    - Sessions can be configured to automatically **unload** encoders after use to minimize VRAM footprint (critical for 16GB/32GB Macs).
    - `GPUCachePolicy` provides high-level presets (e.g., `.lowMemory`, `.highMemory`) for MLX cache limits.

## Dependencies
- `QwenImage` (Core).
- `MLX`, `Logging`.
