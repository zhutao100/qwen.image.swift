import Foundation
import MLX

/// Utility for packing and unpacking latents for layer-aware processing.
/// Used by the layered image generation pipeline.
public enum LatentPacking {

  /// Pack layer latents: [B, L, C, H, W] -> [B, L*(H/2)*(W/2), C*4]
  /// Applies 2x2 patch packing to each layer.
  public static func pack(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    precondition(shape.count == 5, "Expected 5D tensor [B, L, C, H, W], got \(shape.count)D")

    let b = shape[0]
    let l = shape[1]
    let c = shape[2]
    let h = shape[3]
    let w = shape[4]

    precondition(h % 2 == 0, "Height must be divisible by 2, got \(h)")
    precondition(w % 2 == 0, "Width must be divisible by 2, got \(w)")

    // Reshape: [B, L, C, H/2, 2, W/2, 2]
    let reshaped = x.reshaped([b, l, c, h / 2, 2, w / 2, 2])

    // Permute: [B, L, H/2, W/2, C, 2, 2] -> axes [0, 1, 3, 5, 2, 4, 6]
    let permuted = reshaped.transposed(0, 1, 3, 5, 2, 4, 6)

    return permuted.reshaped([b, l * (h / 2) * (w / 2), c * 4])
  }

  /// Unpack to layer latents: [B, patches, C*4] -> [B, L+1, C, H, W]
  /// Reverses the 2x2 patch packing.
  public static func unpack(_ x: MLXArray, layers: Int, height: Int, width: Int) -> MLXArray {
    let shape = x.shape
    precondition(shape.count == 3, "Expected 3D tensor [B, patches, channels], got \(shape.count)D")

    let b = shape[0]
    let c = 16  // Latent channels (z_dim)

    precondition(height % 2 == 0, "Height must be divisible by 2, got \(height)")
    precondition(width % 2 == 0, "Width must be divisible by 2, got \(width)")

    let halfH = height / 2
    let halfW = width / 2
    let expectedPatches = (layers + 1) * halfH * halfW
    precondition(
      shape[1] == expectedPatches,
      "Expected \(expectedPatches) patches, got \(shape[1])"
    )
    precondition(shape[2] == c * 4, "Expected \(c * 4) channels, got \(shape[2])")

    let reshaped = x.reshaped([b, layers + 1, halfH, halfW, c, 2, 2])
    let permuted = reshaped.transposed(0, 1, 4, 2, 5, 3, 6)
    return permuted.reshaped([b, layers + 1, c, height, width])
  }

  /// Pack a single image: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
  /// Convenience for packing the condition image.
  public static func packSingle(_ x: MLXArray) -> MLXArray {
    let shape = x.shape
    precondition(shape.count == 4, "Expected 4D tensor [B, C, H, W], got \(shape.count)D")

    let withLayer = x.reshaped([shape[0], 1, shape[1], shape[2], shape[3]])
    return pack(withLayer)
  }

  /// Calculate the packed sequence length for given parameters.
  public static func packedSequenceLength(layers: Int, height: Int, width: Int) -> Int {
    return (layers + 1) * (height / 2) * (width / 2)
  }
}
