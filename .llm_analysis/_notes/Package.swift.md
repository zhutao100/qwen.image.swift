# Package.swift Analysis

## Overview
Defines the `qwen.image.swift` package structure, platforms, and dependencies.

## Platforms
- macOS v14+
- iOS v16+

## Products
- **QwenImage** (Library): The core neural network and pipeline logic.
- **QwenImageRuntime** (Library): *New* middleware layer for caching and policy.
- **QwenImageCLI** (Executable): Command-line interface.
- **QwenImageApp** (Executable): *New* SwiftUI GUI application.

## Targets

### 1. QwenImage
- **Type**: Library (Core)
- **Dependencies**:
    - `mlx-swift` (Full suite: MLX, NN, Random, Fast, Optimizers)
    - `swift-transformers`
    - `swift-log`

### 2. QwenImageRuntime
- **Type**: Library (Middleware)
- **Dependencies**:
    - `QwenImage` (Internal)
    - `MLX`
    - `Logging`

### 3. QwenImageCLI
- **Type**: Executable
- **Dependencies**:
    - `QwenImage`
    - `QwenImageRuntime`

### 4. QwenImageApp
- **Type**: Executable
- **Dependencies**:
    - `QwenImage`
    - `QwenImageRuntime`

## Key Observations
- The project structure has evolved from a simple Lib+CLI to a layered architecture: `Core -> Runtime -> App/CLI`.
- `mlx-swift` version pinned to `0.29.1` (up to next minor).