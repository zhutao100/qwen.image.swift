# Module: QwenImageApp

## Purpose
The **QwenImageApp** is a full-featured macOS GUI application built with SwiftUI. It provides an interactive environment for all Qwen-Image capabilities, including text-to-image, editing, and layered decomposition.

## Key Files
- `Sources/QwenImageApp/QwenImageApp.swift`: App entry point and global configuration.
- `Sources/QwenImageApp/AppState.swift`: Global state container (Observation framework).
- `Sources/QwenImageApp/Services/ModelService.swift`: Singleton actor managing model downloads and runtime sessions.
- `Sources/QwenImageApp/ViewModels/`: ViewModels for specific features (TextToImage, Layered, Editing).
- `Sources/QwenImageApp/Views/`: SwiftUI views.

## Architecture
- **MVVM (Model-View-ViewModel):**
    - **Model:** `QwenImageRuntime` Sessions and `ModelDefinition` structs.
    - **ViewModel:** `@Observable` classes (`TextToImageViewModel`) that expose state to Views and call into Services.
    - **View:** SwiftUI views that bind to ViewModels.
- **Service Layer:**
    - `ModelService`: A central Actor that manages the lifecycle of `QwenImageRuntime` sessions. It handles model downloading (`HubSnapshot`), loading/unloading pipelines, and switching between modes (Layered vs. Edit).
- **State Management:**
    - `AppState`: The single source of truth for global app state (selected sidebar item, download progress, error reporting). Injected into the SwiftUI Environment.
- **Concurrency:**
    - Heavy operations (generation, loading) are performed in `Task.detached` blocks to keep the UI responsive.
    - Updates to UI state are marshaled back to the `@MainActor`.

## Features
- **Sidebar Navigation:** Switch between Layered, T2I, and Editing modes.
- **Model Manager:** UI for downloading and managing different model variants (Quantized vs Full).
- **Progress Tracking:** Real-time progress bars for downloads and generation steps.
- **Settings:** configuration for default parameters, memory policies, and more.

## Dependencies
- `QwenImage`, `QwenImageRuntime`.
- `SwiftUI`, `AppKit`.
