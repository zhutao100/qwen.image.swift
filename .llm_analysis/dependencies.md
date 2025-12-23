# Project Dependencies

## External Libraries (via SwiftPM)
- **mlx-swift:** Apple's machine learning framework for Apple Silicon. Used for tensor operations, neural network layers, and optimization.
    - Modules: `MLX`, `MLXNN`, `MLXRandom`, `MLXFast`, `MLXOptimizers`.
- **swift-transformers:** Hugging Face's transformers library port. Used for tokenization (`HubSnapshot`) and model downloading/management.
    - Modules: `Transformers`.
- **swift-log:** Apple's logging API.

## System Frameworks
- **SwiftUI**: User interface framework for `QwenImageApp`.
- **Metal:** Used for GPU acceleration detection.
- **CoreGraphics:** Essential for image processing (pixel data manipulation, format conversion).
- **AppKit** (macOS) / **UIKit** (iOS): Used for high-level image loading and saving (`NSImage`, `UIImage`).
- **Foundation**: Basic system services (FileManager, URL, etc.).
- **Dispatch**: Concurrency.
