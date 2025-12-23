# Qwen-Image Swift Architecture

## Executive Summary
**Qwen-Image Swift** is a high-performance, native implementation of the Qwen-Image family of diffusion models (specifically focused on `Qwen-Image-Edit` and `Qwen-Image-Layered`) for Apple Silicon. It leverages Apple's **MLX** framework to run large-scale diffusion transformers efficiently on macOS and iOS devices.

The project supports text-to-image generation, image editing (inpainting/instruction-based editing), and layered image decomposition. It features a sophisticated **Joint Transformer** architecture where text and visual tokens interact deeply, enabled by a multimodal text encoder and a 3D-aware diffusion backbone.

As of recent updates, the project now includes a **GUI Application** (`QwenImageApp`) and a dedicated **Runtime Layer** (`QwenImageRuntime`) to manage state, caching, and resource policies.

## Technology Stack
- **Language:** Swift 5.9+
- **Machine Learning Core:** [MLX Swift](https://github.com/ml-explore/mlx-swift) (Tensor operations, NN layers, Optimization)
- **Model Management:** [Swift Transformers](https://github.com/huggingface/swift-transformers) (Tokenizers, Hub downloads)
- **UI Framework:** SwiftUI (macOS)
- **Image Processing:** CoreGraphics (Native macOS/iOS image I/O), MLX-based Vectorized Ops
- **Logging:** Swift Log
- **Hardware Acceleration:** Metal (via MLX)

## Architecture Overview
The system follows a **Latent Diffusion** pipeline architecture with a **Flow Matching** scheduler. Uniquely, it employs a **Multimodal Diffusion Transformer (MM-DiT)** design.

### High-Level Style
- **Monolithic Library (`QwenImage`):** The core neural network logic and deterministic pipelines.
- **Runtime Middleware (`QwenImageRuntime`):** A policy-aware layer that manages caching, session state, and resource lifecycle (loading/unloading).
- **Consumers:**
    - **CLI (`QwenImageCLI`):** Command-line tool for batch processing and testing.
    - **App (`QwenImageApp`):** macOS GUI for interactive usage.
- **Lazy Loading:** Components are loaded from disk/network on-demand to minimize memory footprint.

## Component Map

### 1. Entry Points
- **App (`QwenImageApp`):** A full-featured SwiftUI application. Features MVVM architecture, Sidebar navigation, and "Model Service" to manage background generation.
- **CLI (`QwenImageCLI`):** Handles argument parsing, model download/caching, and pipeline orchestration. Supports multiple modes: Generation, Editing, Layered Decomposition, and Quantization.

### 2. Runtime Layer (`QwenImageRuntime`)
*New in this version.*
- **Sessions (`ImagePipelineSession`, `LayeredPipelineSession`):** Actor-based wrappers around core pipelines. They enforce policies like "Release Text Encoder after encoding" to save VRAM.
- **Caching:**
    - `PromptEmbeddingsCache`: Caches text encoder outputs based on model ID, prompt, and quantization.
    - `LayeredCaptionCache`: Caches caption embeddings for layered generation, using **image hashing** to key conditioning on specific input images without re-processing.
- **GPU Policy (`GPUCachePolicy`):** Manages MLX memory limits with presets (e.g., "Low Memory", "High Memory").

### 3. Core Pipeline (`QwenImage.Pipeline`)
- **`QwenImagePipeline`:** The central controller for text-to-image and editing.
    - **Optimization:** Supports "Policy-Free" generation APIs where encoded prompts are passed in, allowing the runtime to manage the encoder lifecycle.
- **`QwenLayeredPipeline`:** Dedicated pipeline for decomposing images into layers.
    - **True CFG:** Implements "True Classifier-Free Guidance" (double forward pass with negative prompt).
- **`QwenFlowMatchScheduler`:** Implements the noise schedule with Sigma Shifting.

### 4. Model Components (`QwenImage.Model`)
- **Transformer (The "UNet"):**
    - A **Joint Transformer** where text and image tokens are concatenated.
    - Uses **AdaLN-Continuous** and **3D RoPE** (Time, Height, Width).
- **Text Encoder:**
    - **Vectorized:** Recent updates vectorized the embedding injection and masking logic for significant performance gains.
    - **Multimodal:** Injects vision tokens into the text sequence.
- **VAE (Variational Autoencoder):**
    - 3D Causal Convolutions.
- **Vision Tower:**
    - Encodes reference images into patch embeddings.

### 5. Data Processing (`QwenImage.Processor`)
- **Vision Preprocessor:**
    - **Vectorized:** Image normalization (Mean/Std) is now fully vectorized in MLX, removing CPU bottlenecks.

### 6. Infrastructure (`QwenImage.Weights`, `QwenImage.Quantization`)
- **Weights Loader:** Maps Hugging Face SafeTensors keys to internal Swift module paths.
- **Quantization:** Supports runtime and offline quantization.
- **LoRA:** Supports dynamic loading of adapters. The CLI now supports complex selectors like `repo:filename.safetensors` or direct HF URLs.

## Data Flow (App Generation Example)

1.  **User Action:** User clicks "Generate" in `TextToImageViewModel`.
2.  **Session:** `ModelService` provides an `ImagePipelineSession`.
3.  **Caching:** Session checks `PromptEmbeddingsCache`.
    - *Miss:* Calls `pipeline.encodeGuidancePrompts`. Text Encoder runs (vectorized). Result is cached.
    - *Hit:* returns cached embedding.
4.  **Resource Management:** Session optionally **unloads** the Text Encoder to free ~7GB VRAM.
5.  **Generation:** Session calls `pipeline.generatePixels`.
    - Scheduler steps through the denoising loop on GPU.
    - `MLX.eval` is only called if a progress listener is attached.
6.  **Display:** Final result is decoded by VAE and displayed in the View.

## Directory Structure
```
Sources/
├── QwenImageApp/       # SwiftUI Application (Views, ViewModels, Services)
├── QwenImageCLI/       # Command Line Interface
├── QwenImageRuntime/   # [NEW] Middleware (Caching, Sessions, Policies)
└── QwenImage/          # Core Library
    ├── Model/          # Neural Network definitions
    ├── Pipeline/       # Orchestration & Scheduling
    ├── Processor/      # Image preprocessing
    ├── Tokenizer/      # Text tokenization
    ├── Weights/        # Loading & Mapping (SafeTensors)
    ├── Quantization/   # Weight quantization logic
    ├── Support/        # Logging & Config
    └── Util/           # Math & Image helpers
```