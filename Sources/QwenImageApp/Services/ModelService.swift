import Foundation
import QwenImage
import QwenImageRuntime
import MLX

private enum LoadedPipeline {
  case layered
  case imageEditing
}

actor ModelService {
  static let shared = ModelService()

  // MARK: - Session Storage

  private var layeredSession: LayeredPipelineSession?
  private var imageSession: ImagePipelineSession?

  // MARK: - Legacy Pipeline Storage (for direct access if needed)

  private var layeredPipeline: QwenLayeredPipeline?
  private var imagePipeline: QwenImagePipeline?

  private var cachedPaths: [String: URL] = [:]
  private var loadedMode: LoadedPipeline?

  private var currentModelId: String?
  private var currentRevision: String = "main"

  private var appliedLayeredLoRA: (url: URL, scale: Float)?

  private var appliedImageLoRA: (url: URL, scale: Float)?

  private init() {
    let preset = GPUCachePolicy.recommendedPreset()
    GPUCachePolicy.applyPreset(preset)
  }

  // MARK: - Path Resolution

  func cachedModelPath(repoId: String) -> URL? {
    if let cached = cachedPaths[repoId] {
      return cached
    }

    let env = ProcessInfo.processInfo.environment
    let hubPath: URL
    if let hfHubCache = env["HF_HUB_CACHE"], !hfHubCache.isEmpty {
      hubPath = URL(fileURLWithPath: hfHubCache)
    } else if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
      hubPath = URL(fileURLWithPath: hfHome).appending(path: "hub")
    } else {
      hubPath = URL(fileURLWithPath: NSHomeDirectory()).appending(path: ".cache/huggingface/hub")
    }

    if let snapshotDir = resolveHuggingFaceCliSnapshotURL(
      hubPath: hubPath,
      repoId: repoId,
      revision: currentRevision,
      fileManager: FileManager.default
    ), isValidModelSnapshot(snapshotDir) {
      cachedPaths[repoId] = snapshotDir
      return snapshotDir
    }

    let modelDir = hubPath.appending(path: "models").appending(path: repoId)

    guard isValidModelSnapshot(modelDir) else {
      return nil
    }

    cachedPaths[repoId] = modelDir
    return modelDir
  }

  private func isValidModelSnapshot(_ url: URL) -> Bool {
    FileManager.default.fileExists(atPath: url.appending(path: "model_index.json").path)
  }

  private func resolveHuggingFaceCliSnapshotURL(
    hubPath: URL,
    repoId: String,
    revision: String,
    fileManager: FileManager
  ) -> URL? {
    // huggingface-cli (huggingface_hub) cache layout:
    //   <HF_HUB_CACHE>/
    //     models--<org>--<repo>/
    //       refs/<revision>          (contains commit hash)
    //       snapshots/<commitHash>/  (materialized snapshot)
    let repoDirectoryName = "models--" + repoId.replacingOccurrences(of: "/", with: "--")
    let repoDirectory = hubPath.appending(path: repoDirectoryName)

    var isRepoDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: repoDirectory.path, isDirectory: &isRepoDirectory),
          isRepoDirectory.boolValue
    else {
      return nil
    }

    let commitHash: String?
    if isCommitHash(revision) {
      commitHash = revision.lowercased()
    } else {
      let refFile = repoDirectory
        .appending(path: "refs")
        .appending(path: revision)
      guard fileManager.fileExists(atPath: refFile.path),
            let contents = try? String(contentsOf: refFile, encoding: .utf8)
      else {
        return nil
      }
      let hash = contents.trimmingCharacters(in: .whitespacesAndNewlines)
      commitHash = isCommitHash(hash) ? hash.lowercased() : nil
    }

    guard let commitHash else { return nil }

    let snapshotDirectory = repoDirectory
      .appending(path: "snapshots")
      .appending(path: commitHash)

    var isSnapshotDirectory: ObjCBool = false
    guard fileManager.fileExists(atPath: snapshotDirectory.path, isDirectory: &isSnapshotDirectory),
          isSnapshotDirectory.boolValue
    else {
      return nil
    }

    return snapshotDirectory
  }

  private func isCommitHash(_ value: String) -> Bool {
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    guard trimmed.count == 40 else { return false }
    let hexSet = CharacterSet(charactersIn: "0123456789abcdefABCDEF")
    return trimmed.unicodeScalars.allSatisfy { hexSet.contains($0) }
  }

  // MARK: - Download

  func downloadModel(
    repoId: String,
    revision: String = "main",
    progressHandler: @escaping @Sendable (HubSnapshotProgress) -> Void
  ) async throws -> URL {
    let options = HubSnapshotOptions(
      repoId: repoId,
      revision: revision
    )

    let snapshot = try HubSnapshot(options: options)
    let path = try await snapshot.prepare(progressHandler: progressHandler)

    cachedPaths[repoId] = path
    return path
  }

  // MARK: - Session-Based Loading

  func loadLayeredSession(
    from path: URL,
    modelId: String = "Qwen/Qwen-Image-Layered",
    revision: String = "main",
    configuration: LayeredPipelineSessionConfiguration = .default
  ) async throws -> LayeredPipelineSession {
    if loadedMode == .imageEditing {
      unloadImagePipeline()
    }

    if let session = layeredSession {
      loadedMode = .layered
      return session
    }

    let pipeline = try await QwenLayeredPipeline.load(from: path, dtype: .bfloat16)
    layeredPipeline = pipeline
    currentModelId = modelId
    currentRevision = revision

    let session = LayeredPipelineSession(
      pipeline: pipeline,
      modelId: modelId,
      revision: revision,
      configuration: configuration
    )
    layeredSession = session
    loadedMode = .layered

    return session
  }

  func loadImageSession(
    from path: URL,
    config: QwenImageConfig,
    modelId: String = "Qwen/Qwen-Image",
    revision: String = "main",
    configuration: ImagePipelineSessionConfiguration = .default
  ) async throws -> ImagePipelineSession {
    if loadedMode == .layered {
      unloadLayeredPipeline()
    }

    if let session = imageSession {
      loadedMode = .imageEditing
      return session
    }

    let pipeline = QwenImagePipeline(config: config)
    pipeline.setBaseDirectory(path)
    try pipeline.prepareTokenizer(from: path, maxLength: nil)
    try pipeline.prepareTextEncoder(from: path)
    try pipeline.prepareVAE(from: path)

    imagePipeline = pipeline
    currentModelId = modelId
    currentRevision = revision

    let session = ImagePipelineSession(
      pipeline: pipeline,
      modelId: modelId,
      revision: revision,
      configuration: configuration
    )
    imageSession = session
    loadedMode = .imageEditing

    return session
  }

  // MARK: - Legacy Pipeline Loading (for backwards compatibility)

  func loadLayeredPipeline(from path: URL) async throws -> QwenLayeredPipeline {
    if loadedMode == .imageEditing {
      unloadImagePipeline()
    }

    if let cached = layeredPipeline {
      loadedMode = .layered
      return cached
    }

    let pipeline = try await QwenLayeredPipeline.load(from: path, dtype: .bfloat16)
    layeredPipeline = pipeline
    loadedMode = .layered
    return pipeline
  }

  func loadImagePipeline(from path: URL, config: QwenImageConfig) async throws -> QwenImagePipeline {
    if loadedMode == .layered {
      unloadLayeredPipeline()
    }

    if let cached = imagePipeline {
      loadedMode = .imageEditing
      return cached
    }

    let pipeline = QwenImagePipeline(config: config)
    pipeline.setBaseDirectory(path)

    try pipeline.prepareTokenizer(from: path, maxLength: nil)
    try pipeline.prepareTextEncoder(from: path)
    try pipeline.prepareVAE(from: path)

    imagePipeline = pipeline
    loadedMode = .imageEditing
    return pipeline
  }

  // MARK: - Unloading

  func unloadLayeredPipeline() {
    layeredSession = nil
    layeredPipeline = nil
    appliedLayeredLoRA = nil
    if loadedMode == .layered {
      loadedMode = nil
      currentModelId = nil
    }
    GPUCachePolicy.clearCache()
  }

  func unloadImagePipeline() {
    imageSession = nil
    imagePipeline = nil
    appliedImageLoRA = nil
    if loadedMode == .imageEditing {
      loadedMode = nil
      currentModelId = nil
    }
    GPUCachePolicy.clearCache()
  }

  func unloadAll() {
    layeredSession = nil
    imageSession = nil
    layeredPipeline = nil
    imagePipeline = nil
    appliedLayeredLoRA = nil
    appliedImageLoRA = nil
    loadedMode = nil
    currentModelId = nil
    GPUCachePolicy.clearCache()
  }

  // MARK: - LoRA

  @discardableResult
  func applyLoRA(
    to pipeline: QwenLayeredPipeline,
    from url: URL,
    scale: Float
  ) throws -> Bool {
    if let applied = appliedLayeredLoRA,
       applied.url == url,
       applied.scale == scale {
      return false
    }

    try pipeline.applyLora(from: url, scale: scale)
    appliedLayeredLoRA = (url: url, scale: scale)
    return true
  }

  func applyLoRAToLayeredSession(from url: URL, scale: Float) async throws {
    guard let session = layeredSession else {
      throw ModelServiceError.noSessionLoaded
    }
    try await session.applyLora(from: url, scale: scale)
  }

  // MARK: - Cache Management

  func clearAllCaches() async {
    if let session = layeredSession {
      await session.clearCache()
    }
    if let session = imageSession {
      await session.clearCache()
    }
  }

  // MARK: - Status

  var isLayeredLoaded: Bool {
    layeredPipeline != nil
  }

  var isImageLoaded: Bool {
    imagePipeline != nil
  }

  var hasLayeredSession: Bool {
    layeredSession != nil
  }

  var hasImageSession: Bool {
    imageSession != nil
  }
}

// MARK: - Errors

enum ModelServiceError: Error {
  case noSessionLoaded
  case pipelineNotLoaded
}

// MARK: - HubSnapshotProgress Extensions

extension HubSnapshotProgress {
  var formattedSpeed: String? {
    guard let speed = estimatedSpeedBytesPerSecond else { return nil }
    return ByteCountFormatter.string(fromByteCount: Int64(speed), countStyle: .file) + "/s"
  }

  var formattedCompleted: String {
    ByteCountFormatter.string(fromByteCount: completedUnitCount, countStyle: .file)
  }

  var formattedTotal: String {
    totalUnitCount > 0
      ? ByteCountFormatter.string(fromByteCount: totalUnitCount, countStyle: .file)
      : "Unknown"
  }

  var formattedTimeRemaining: String? {
    guard let speed = estimatedSpeedBytesPerSecond, speed > 0 else { return nil }
    let remainingBytes = totalUnitCount - completedUnitCount
    let secondsRemaining = Double(remainingBytes) / Double(speed)

    if secondsRemaining < 60 {
      return "~\(Int(secondsRemaining))s remaining"
    } else if secondsRemaining < 3600 {
      return "~\(Int(secondsRemaining / 60))m remaining"
    } else {
      let hours = Int(secondsRemaining / 3600)
      let minutes = Int((secondsRemaining.truncatingRemainder(dividingBy: 3600)) / 60)
      return "~\(hours)h \(minutes)m remaining"
    }
  }
}
