import Foundation
import QwenImage
import MLX
import CryptoKit

// MARK: - Image Hash Utilities

/// Utilities for computing image hashes from original file data.
///
/// Unlike the core pipeline's MLX-based hash (which requires MLX.eval),
/// these utilities compute hashes from the original image bytes before
/// any MLX conversion occurs.
public enum ImageHashUtility {
  /// Compute a SHA256 hash from image file data.
  /// - Parameter data: The raw image file data (PNG, JPEG, etc.)
  /// - Returns: A hex string representation of the hash.
  public static func sha256Hash(from data: Data) -> String {
    let hash = SHA256.hash(data: data)
    return hash.compactMap { String(format: "%02x", $0) }.joined()
  }

  /// Compute a fast hash from image file data.
  /// Uses first/last bytes and size for a quick (but less collision-resistant) hash.
  /// - Parameter data: The raw image file data.
  /// - Returns: A hash string suitable for cache keys.
  public static func fastHash(from data: Data) -> String {
    guard !data.isEmpty else { return "empty" }

    var hasher = Hasher()

    // Include file size
    hasher.combine(data.count)

    // Include first 1KB
    let prefixCount = min(1024, data.count)
    data.prefix(prefixCount).forEach { hasher.combine($0) }

    // Include last 1KB if file is large enough
    if data.count > 2048 {
      let suffixStart = data.count - 1024
      data.suffix(from: suffixStart).forEach { hasher.combine($0) }
    }

    let hashValue = hasher.finalize()
    return String(format: "%016llx", UInt64(bitPattern: Int64(hashValue)))
  }

  /// Load image data from a file URL.
  /// - Parameter url: The file URL.
  /// - Returns: The file data, or nil if loading fails.
  public static func loadImageData(from url: URL) -> Data? {
    try? Data(contentsOf: url)
  }
}

// MARK: - Cache Key

/// Cache key for layered caption embeddings.
///
/// This key captures the factors that affect caption encoding:
/// - Model identity
/// - Prompt text
/// - Image identity (via hash of original bytes)
public struct LayeredCaptionCacheKey: Hashable, Sendable {
  /// Model identifier (repo ID or local path)
  public let modelId: String

  /// Model revision/commit hash
  public let revision: String

  /// The prompt text
  public let prompt: String

  /// The negative prompt text (if any)
  public let negativePrompt: String?

  /// Hash of original image bytes (computed BEFORE MLX conversion)
  public let imageHash: String

  public init(
    modelId: String,
    revision: String = "main",
    prompt: String,
    negativePrompt: String? = nil,
    imageHash: String
  ) {
    self.modelId = modelId
    self.revision = revision
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.imageHash = imageHash
  }

  /// Create a cache key using image file data for hashing.
  /// - Parameters:
  ///   - modelId: Model identifier.
  ///   - revision: Model revision.
  ///   - prompt: Prompt text.
  ///   - negativePrompt: Optional negative prompt.
  ///   - imageData: Raw image file data (used to compute hash).
  ///   - useFastHash: If true, uses faster but less collision-resistant hash.
  public init(
    modelId: String,
    revision: String = "main",
    prompt: String,
    negativePrompt: String? = nil,
    imageData: Data,
    useFastHash: Bool = false
  ) {
    self.modelId = modelId
    self.revision = revision
    self.prompt = prompt
    self.negativePrompt = negativePrompt
    self.imageHash = useFastHash
      ? ImageHashUtility.fastHash(from: imageData)
      : ImageHashUtility.sha256Hash(from: imageData)
  }
}

// MARK: - Cache Entry

/// A cached layered caption encoding with metadata.
struct LayeredCaptionCacheEntry {
  let encoding: LayeredPromptEncoding
  let negativeEncoding: LayeredPromptEncoding?
  let createdAt: Date
  let accessCount: Int

  init(encoding: LayeredPromptEncoding, negativeEncoding: LayeredPromptEncoding? = nil) {
    self.encoding = encoding
    self.negativeEncoding = negativeEncoding
    self.createdAt = Date()
    self.accessCount = 1
  }

  func incrementingAccessCount() -> LayeredCaptionCacheEntry {
    LayeredCaptionCacheEntry(
      encoding: encoding,
      negativeEncoding: negativeEncoding,
      createdAt: createdAt,
      accessCount: accessCount + 1
    )
  }

  private init(
    encoding: LayeredPromptEncoding,
    negativeEncoding: LayeredPromptEncoding?,
    createdAt: Date,
    accessCount: Int
  ) {
    self.encoding = encoding
    self.negativeEncoding = negativeEncoding
    self.createdAt = createdAt
    self.accessCount = accessCount
  }
}

// MARK: - Cache

/// Thread-safe cache for layered caption embeddings.
///
/// This cache stores precomputed caption embeddings for layered generation.
/// The cache key uses a hash of the original image bytes (computed BEFORE
/// MLX conversion) to avoid forcing MLX.eval() just for cache key computation.
///
/// Usage:
/// ```swift
/// let cache = LayeredCaptionCache(maxEntries: 5)
///
/// // Load image data for hashing (before MLX conversion)
/// let imageData = try Data(contentsOf: imageURL)
///
/// let key = LayeredCaptionCacheKey(
///   modelId: "Qwen/Qwen-Image-Layered",
///   prompt: "foreground and background",
///   imageData: imageData
/// )
///
/// if let cached = await cache.get(key: key) {
///   // Use cached encoding
/// } else {
///   let encoding = try pipeline.encodePromptToEncoding(prompt, dtype: .bfloat16)
///   await cache.set(key: key, encoding: encoding)
/// }
/// ```
public actor LayeredCaptionCache {
  private var cache: [LayeredCaptionCacheKey: LayeredCaptionCacheEntry] = [:]
  private let maxEntries: Int

  /// Create a new cache with the specified maximum number of entries.
  /// - Parameter maxEntries: Maximum number of cached embeddings. Default is 5.
  public init(maxEntries: Int = 5) {
    self.maxEntries = max(1, maxEntries)
  }

  /// Get cached encodings for the given key.
  /// - Parameter key: The cache key.
  /// - Returns: Tuple of (positive encoding, negative encoding) or nil if not found.
  public func get(key: LayeredCaptionCacheKey) -> (LayeredPromptEncoding, LayeredPromptEncoding?)? {
    guard let entry = cache[key] else { return nil }
    cache[key] = entry.incrementingAccessCount()
    return (entry.encoding, entry.negativeEncoding)
  }

  /// Cache encodings for the given key.
  /// - Parameters:
  ///   - key: The cache key.
  ///   - encoding: The positive prompt encoding.
  ///   - negativeEncoding: Optional negative prompt encoding.
  public func set(
    key: LayeredCaptionCacheKey,
    encoding: LayeredPromptEncoding,
    negativeEncoding: LayeredPromptEncoding? = nil
  ) {
    if cache.count >= maxEntries && cache[key] == nil {
      evictLeastRecentlyUsed()
    }
    cache[key] = LayeredCaptionCacheEntry(encoding: encoding, negativeEncoding: negativeEncoding)
  }

  /// Remove a specific cached encoding.
  /// - Parameter key: The cache key to invalidate.
  public func invalidate(key: LayeredCaptionCacheKey) {
    cache.removeValue(forKey: key)
  }

  /// Remove all cached encodings.
  public func invalidateAll() {
    cache.removeAll()
  }

  /// Remove all cached encodings for a specific model.
  /// - Parameter modelId: The model ID to invalidate.
  public func invalidateForModel(_ modelId: String) {
    let keysToRemove = cache.keys.filter { $0.modelId == modelId }
    for key in keysToRemove {
      cache.removeValue(forKey: key)
    }
  }

  /// Remove all cached encodings for a specific image.
  /// - Parameter imageHash: The image hash to invalidate.
  public func invalidateForImage(_ imageHash: String) {
    let keysToRemove = cache.keys.filter { $0.imageHash == imageHash }
    for key in keysToRemove {
      cache.removeValue(forKey: key)
    }
  }

  /// The current number of cached entries.
  public var count: Int {
    cache.count
  }

  /// Check if a key is in the cache.
  public func contains(key: LayeredCaptionCacheKey) -> Bool {
    cache[key] != nil
  }

  // MARK: - Private

  private func evictLeastRecentlyUsed() {
    guard let lruKey = cache.min(by: { a, b in
      if a.value.accessCount != b.value.accessCount {
        return a.value.accessCount < b.value.accessCount
      }
      return a.value.createdAt < b.value.createdAt
    })?.key else {
      return
    }
    cache.removeValue(forKey: lruKey)
  }
}
