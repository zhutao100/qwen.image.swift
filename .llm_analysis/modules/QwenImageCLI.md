# Module: QwenImageCLI

## Purpose
The Command Line Interface (CLI) entry point for the `qwen.image.swift` project. It exposes functionality for text-to-image generation, image editing, layered image decomposition, and model quantization.

## Key Files
- `Sources/QwenImageCLI/main.swift`: The single file containing the CLI logic, argument parsing, and execution flow.
- `Sources/QwenImageCLI/LoraReference.swift`: Parser for complex LoRA specifiers.

## Functionality
- **Argument Parsing:** Manually parses `CommandLine.arguments` to configure generation parameters (prompt, steps, dimensions, seeds, etc.).
- **Device Management:** Configures MLX GPU cache limits using `QwenImageRuntime.GPUCachePolicy`.
- **Logging:** Implements a custom `FileLogHandler` and bootstraps `swift-log` to support file and stderr logging.
- **Execution Modes:**
    1.  **Text-to-Image:** Default mode. Uses `QwenImagePipeline` to generate images from prompts.
    2.  **Image Editing:** Activated via `--reference-image`. Uses `QwenImagePipeline` with `.imageEditing` config.
    3.  **Layered Generation:** Activated via `--layered`. Uses `QwenLayeredPipeline` to decompose images into layers.
    4.  **Quantization:**
        - **Pure Swift:** Uses `SwiftQuantSaver` to quantize and save weights entirely in Swift. Python dependencies have been removed.
- **LoRA Support:**
    - Supports advanced LoRA references: `repo:filename.safetensors` or direct HF URLs via `LoraReferenceParser`.

## Dependencies
- **Internal:** `QwenImage` (Core library), `QwenImageRuntime` (Policy/Caching).
- **External:** `MLX`, `MLXNN`, `MLXRandom`, `Logging` (swift-log).
- **System:** `Foundation`, `Dispatch`, `Metal`, `CoreGraphics`, `AppKit`/`UIKit`.

## Usage
Run via `swift run QwenImageCLI [arguments]`.
Example: `swift run QwenImageCLI --prompt "A futuristic city" --steps 20`