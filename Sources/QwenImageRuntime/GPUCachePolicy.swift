import Foundation
import MLX

// MARK: - GPU Cache Policy

/// GPU cache management utilities.
///
/// These are runtime-layer concerns for managing MLX GPU memory.
/// The core pipelines do not set GPU cache limits - that's a policy
/// decision for the runtime/app layer.
///
/// Usage:
/// ```swift
/// // Set a 4GB cache limit
/// GPUCachePolicy.setCacheLimit(4 * 1024 * 1024 * 1024)
///
/// // Or use a preset
/// GPUCachePolicy.applyPreset(.highMemory)
///
/// // Clear the cache when needed
/// GPUCachePolicy.clearCache()
/// ```
public enum GPUCachePolicy {
  // MARK: - Presets

  /// Predefined cache limit presets for common scenarios.
  public enum Preset: Sendable {
    /// Minimal cache (2GB) for memory-constrained environments.
    case lowMemory

    /// Standard cache (4GB) for typical usage.
    case standard

    /// Large cache (8GB) for high-performance workloads.
    case highMemory

    /// Very large cache (16GB) for systems with abundant memory.
    case veryHighMemory

    /// Maximum cache (64GB) for high-end workstations.
    case maximum

    /// No limit - let MLX manage memory automatically.
    case unlimited

    /// Custom cache limit in bytes.
    case custom(Int)

    /// The cache limit in bytes for this preset.
    public var bytes: Int? {
      switch self {
      case .lowMemory:
        return 2 * 1024 * 1024 * 1024  // 2GB
      case .standard:
        return 4 * 1024 * 1024 * 1024  // 4GB
      case .highMemory:
        return 8 * 1024 * 1024 * 1024  // 8GB
      case .veryHighMemory:
        return 16 * 1024 * 1024 * 1024  // 16GB
      case .maximum:
        return 64 * 1024 * 1024 * 1024  // 64GB
      case .unlimited:
        return nil
      case .custom(let bytes):
        return bytes
      }
    }
  }

  // MARK: - Cache Management

  /// Set the MLX GPU cache limit.
  /// - Parameter bytes: The cache limit in bytes.
  public static func setCacheLimit(_ bytes: Int) {
    GPU.set(cacheLimit: bytes)
  }

  /// Apply a cache limit preset.
  /// - Parameter preset: The preset to apply.
  public static func applyPreset(_ preset: Preset) {
    if let bytes = preset.bytes {
      GPU.set(cacheLimit: bytes)
    }
    // For .unlimited, we don't set a limit (MLX default behavior)
  }

  /// Clear the GPU cache.
  /// This releases cached GPU memory back to the system.
  public static func clearCache() {
    GPU.clearCache()
  }

  // MARK: - Recommendations

  /// Get a recommended cache limit based on a fraction of system memory.
  /// - Parameter fraction: Fraction of system memory to use (0.0 to 1.0).
  /// - Returns: Recommended cache limit in bytes.
  public static func recommendedCacheLimit(fraction: Double = 0.5) -> Int {
    let totalMemory = ProcessInfo.processInfo.physicalMemory
    let recommended = Double(totalMemory) * min(1.0, max(0.1, fraction))
    return Int(recommended)
  }

  /// Get a recommended preset based on system memory.
  /// - Returns: A preset appropriate for the system's memory.
  public static func recommendedPreset() -> Preset {
    let totalMemoryGB = ProcessInfo.processInfo.physicalMemory / (1024 * 1024 * 1024)

    switch totalMemoryGB {
    case 0..<16:
      return .lowMemory
    case 16..<32:
      return .standard
    case 32..<64:
      return .highMemory
    case 64..<128:
      return .veryHighMemory
    default:
      return .maximum
    }
  }

  // MARK: - System Info

  /// Total system physical memory in bytes.
  public static var totalSystemMemory: UInt64 {
    ProcessInfo.processInfo.physicalMemory
  }

  /// Total system physical memory as a formatted string.
  public static var totalSystemMemoryFormatted: String {
    ByteCountFormatter.string(fromByteCount: Int64(totalSystemMemory), countStyle: .memory)
  }
}

// MARK: - Convenience Extensions

extension GPUCachePolicy.Preset: CustomStringConvertible {
  public var description: String {
    switch self {
    case .lowMemory:
      return "Low Memory (2GB)"
    case .standard:
      return "Standard (4GB)"
    case .highMemory:
      return "High Memory (8GB)"
    case .veryHighMemory:
      return "Very High Memory (16GB)"
    case .maximum:
      return "Maximum (64GB)"
    case .unlimited:
      return "Unlimited"
    case .custom(let bytes):
      let formatted = ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
      return "Custom (\(formatted))"
    }
  }
}
