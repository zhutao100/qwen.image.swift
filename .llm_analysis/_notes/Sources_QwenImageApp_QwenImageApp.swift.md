# Sources/QwenImageApp/QwenImageApp.swift Analysis

## Purpose
- **Main Entry Point**: The SwiftUI application lifecycle root.
- **Environment**: Sets up the global `AppState` and configures global GPU policies on launch.

## Architecture
- **MVVM**: Uses `AppState` as the central source of truth, injecting it into the environment.
- **Service-Oriented**: Delegates heavy lifting to `ModelService` (singleton actor) which manages `QwenImageRuntime` sessions.
- **Observation**: Uses the Swift 5.9 `@Observable` macro for reactive UI updates.

## Key Files
- `AppState.swift`: Global state (sidebar selection, model status, generation queue).
- `ModelService.swift`: Manages model downloads (`HubSnapshot`), sessions, and loading/unloading logic.
- `TextToImageViewModel.swift`: ViewModel for the T2I feature.

## Quality Assessment
- **Modern Swift**: Uses latest SwiftUI and Swift concurrency features.
- **Robustness**: Handles model downloading, error states, and background generation cleanly.
