import Foundation
import MLX
import MLXNN

public final class QwenVAE: Module {
  static let latentsMean: MLXArray = {
    let values: [Float] = [
      -0.7571, -0.7089, -0.9113, 0.1075,
      -0.1745, 0.9653, -0.1517, 1.5508,
      0.4134, -0.0715, 0.5517, -0.3632,
      -0.1922, -0.9497, 0.2503, -0.2921
    ]
    return MLXArray(values, [1, 16, 1, 1, 1])
  }()

  static let latentsStd: MLXArray = {
    let values: [Float] = [
      2.8184, 1.4541, 2.3275, 2.6558,
      1.2196, 1.7708, 2.6052, 2.0743,
      3.2687, 2.1526, 2.8652, 1.5579,
      1.6382, 1.1253, 2.8251, 1.916
    ]
    return MLXArray(values, [1, 16, 1, 1, 1])
  }()

  @ModuleInfo(key: "encoder") var encoder: QwenImageEncoder3D
  @ModuleInfo(key: "decoder") var decoder: QwenImageDecoder3D
  @ModuleInfo(key: "post_quant_conv") var postQuantConv: QwenImageCausalConv3D
  @ModuleInfo(key: "quant_conv") var quantConv: QwenImageCausalConv3D

  public override init() {
    self._encoder.wrappedValue = QwenImageEncoder3D()
    self._decoder.wrappedValue = QwenImageDecoder3D()
    self._postQuantConv.wrappedValue = QwenImageCausalConv3D(
      inputChannels: 16,
      outputChannels: 16,
      kernelSize: (1, 1, 1),
      stride: (1, 1, 1),
      padding: (0, 0, 0)
    )
    self._quantConv.wrappedValue = QwenImageCausalConv3D(
      inputChannels: 32,
      outputChannels: 32,
      kernelSize: (1, 1, 1),
      stride: (1, 1, 1),
      padding: (0, 0, 0)
    )
    super.init()
  }

  /// Decode latents to image.
  /// Note: The caller should denormalize latents (latent * std + mean) before calling this method.
  /// This matches Python's VAE._decode() behavior where denormalization is done by the pipeline.
  public func decode(_ latents: MLXArray) -> MLXArray {
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let height = latents.dim(2)
    let width = latents.dim(3)

    var hidden = latents
    hidden = hidden.reshaped(batch, channels, 1, height, width)
    // Note: Denormalization (latent * std + mean) is done by the pipeline before calling decode,
    // matching Python behavior where VAE._decode() does not apply denormalization
    hidden = postQuantConv(hidden)
    hidden = decoder(hidden)
    return hidden[0..., 0..., 0, 0..., 0...]
  }

  /// Decode latents with denormalization included (for backward compatibility).
  /// Use this for existing pipelines that expect the old behavior.
  public func decodeWithDenormalization(_ latents: MLXArray) -> MLXArray {
    let batch = latents.dim(0)
    let channels = latents.dim(1)
    let height = latents.dim(2)
    let width = latents.dim(3)

    var hidden = latents
    hidden = hidden.reshaped(batch, channels, 1, height, width)
    hidden = hidden * Self.latentsStd.asType(hidden.dtype) + Self.latentsMean.asType(hidden.dtype)
    hidden = postQuantConv(hidden)
    hidden = decoder(hidden)
    return hidden[0..., 0..., 0, 0..., 0...]
  }

  public func encode(_ images: MLXArray) -> MLXArray {
    let result = encodeWithIntermediates(images)
    return result.latents
  }

  /// Encode images and return normalized latents (for backward compatibility with existing pipelines).
  public func encodeWithIntermediates(_ images: MLXArray) -> (latents: MLXArray, encoderHidden: MLXArray, quantHidden: MLXArray) {
    let (rawLatents, encoderHidden, quantHidden) = encodeRaw(images)

    let mean = Self.latentsMean[0..., 0..., 0, 0..., 0...].asType(rawLatents.dtype)
    let std = Self.latentsStd[0..., 0..., 0, 0..., 0...].asType(rawLatents.dtype)

    let normalized = (rawLatents - mean) / std
    return (normalized, encoderHidden, quantHidden)
  }

  /// Encode images and return raw (unnormalized) latents.
  /// Use this for pipelines that handle normalization themselves (like layered generation).
  public func encodeRaw(_ images: MLXArray) -> (latents: MLXArray, encoderHidden: MLXArray, quantHidden: MLXArray) {
    precondition(images.ndim == 4, "Expected input in NCHW format")
    let batch = images.dim(0)
    let channels = images.dim(1)
    let height = images.dim(2)
    let width = images.dim(3)

    var reshaped = images
    reshaped = reshaped.reshaped(batch, channels, 1, height, width)
    let encoderHidden = encoder(reshaped)
    let quantHidden = quantConv(encoderHidden)

    // Return raw latent (first 16 channels) - normalization is done by the pipeline
    let selected = quantHidden[0..., 0..<16, 0, 0..., 0...]
    return (selected, encoderHidden, quantHidden)
  }

  /// Denormalize a latent array for decoding.
  public static func denormalizeLatent(_ latent: MLXArray) -> MLXArray {
    let mean = latentsMean.asType(latent.dtype)
    let std = latentsStd.asType(latent.dtype)

    if latent.ndim == 4 {
      // [B, C, H, W] -> use squeezed mean/std [1, 16, 1, 1]
      let mean4d = mean[0..., 0..., 0, 0..., 0...]
      let std4d = std[0..., 0..., 0, 0..., 0...]
      return latent * std4d + mean4d
    } else if latent.ndim == 5 {
      // For layered generation: [B, L, C, H, W] - reshape mean/std to [1, 1, 16, 1, 1]
      // Note: mean/std are stored as [1, 16, 1, 1, 1] for [B, C, F, H, W] format
      // So we need to transpose to match [B, L, C, H, W] layout
      let mean5d = mean.transposed(0, 2, 1, 3, 4)  // [1, 16, 1, 1, 1] -> [1, 1, 16, 1, 1]
      let std5d = std.transposed(0, 2, 1, 3, 4)    // [1, 16, 1, 1, 1] -> [1, 1, 16, 1, 1]
      return latent * std5d + mean5d
    } else {
      fatalError("Expected 4D or 5D latent, got \(latent.ndim)D")
    }
  }

  /// Normalize a latent array after encoding.
  public static func normalizeLatent(_ latent: MLXArray) -> MLXArray {
    let mean = latentsMean.asType(latent.dtype)
    let std = latentsStd.asType(latent.dtype)

    if latent.ndim == 4 {
      // [B, C, H, W] -> use squeezed mean/std
      let mean4d = mean[0..., 0..., 0, 0..., 0...]
      let std4d = std[0..., 0..., 0, 0..., 0...]
      return (latent - mean4d) / std4d
    } else if latent.ndim == 5 {
      // [B, L, C, H, W] or [B, C, F, H, W] - use full mean/std
      return (latent - mean) / std
    } else {
      fatalError("Expected 4D or 5D latent, got \(latent.ndim)D")
    }
  }
}
