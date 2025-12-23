# Sources/QwenImageCLI/main.swift Analysis

## Purpose
- Entry point for the Command Line Interface.
- Handles argument parsing, model loading, pipeline execution, and image I/O.

## Key Functions
- `QwenImageCLIEntry.run()`: Main logic flow.
- `printUsage()`: Help message.
- `resolveSnapshot(...)`: Downloads/locates model weights.
- `runLayeredGeneration(...)`: Specific logic for layered image decomposition.
- `cgImageToMLXArray(...)` / `mlxArrayToImage(...)`: Image format conversion.

## Recent Updates
- **Pure Swift**: Removed `runQuantizeSnapshot` (Python) and `rsync` dependencies. Quantization is now fully handled by Swift code (`SwiftQuantSaver`), making the CLI more portable and self-contained.
- **Runtime Integration**: Now uses `GPUCachePolicy` from `QwenImageRuntime` to set memory limits based on system RAM.
- **LoRA Support**: Integrated `LoraReferenceParser` to support advanced LoRA selection (repo:file syntax, HF URLs).

## Quality Observations
- **Complexity**: The file is still large but improved by removing the Python subprocess logic.
- **Duplication**: Argument parsing is manual and verbose. A library like `ArgumentParser` would clean this up significantly.
- **Image Processing**: Contains raw CoreGraphics code for image conversion. This logic seems like it belongs in `QwenImage/Util` or `Processor`.