import Foundation
import MLX
import MLXNN

public final class QwenLayeredRoPE: Module {
  let theta: Float
  let axesDim: [Int]
  let scaleRope: Bool

  // Pre-computed frequency tensors
  var posFreqs: MLXArray
  var negFreqs: MLXArray

  public init(theta: Float = 10000, axesDim: [Int] = [16, 56, 56], scaleRope: Bool = true) {
    self.theta = theta
    self.axesDim = axesDim
    self.scaleRope = scaleRope

    // Pre-compute frequencies for positions 0..4095
    let posIndex = MLXArray(0..<4096)
    let negIndex = MLXArray((0..<4096).reversed().map { Float(-$0 - 1) })

    // Compute frequencies for each axis
    let posFreq0 = Self.ropeParams(index: posIndex, dim: axesDim[0], theta: theta)
    let posFreq1 = Self.ropeParams(index: posIndex, dim: axesDim[1], theta: theta)
    let posFreq2 = Self.ropeParams(index: posIndex, dim: axesDim[2], theta: theta)
    self.posFreqs = MLX.concatenated([posFreq0, posFreq1, posFreq2], axis: 1)

    let negFreq0 = Self.ropeParams(index: negIndex, dim: axesDim[0], theta: theta)
    let negFreq1 = Self.ropeParams(index: negIndex, dim: axesDim[1], theta: theta)
    let negFreq2 = Self.ropeParams(index: negIndex, dim: axesDim[2], theta: theta)
    self.negFreqs = MLX.concatenated([negFreq0, negFreq1, negFreq2], axis: 1)
  }

  private static func ropeParams(index: MLXArray, dim: Int, theta: Float) -> MLXArray {
    precondition(dim % 2 == 0, "Dimension must be even")

    // freqs = 1 / (theta^(2i/dim)) for i in 0..<dim/2
    let exponents = MLXArray(stride(from: 0, to: dim, by: 2).map { Float($0) }) / Float(dim)
    let freqScale = MLX.pow(MLXArray(theta), exponents)
    let invFreq = 1.0 / freqScale

    // Outer product: [seq_len, dim/2]
    let indexFloat = index.asType(.float32)
    let angles = indexFloat[0..., .newAxis] * invFreq[.newAxis, 0...]

    // Return cos and sin stacked: [seq_len, dim/2, 2]
    let cosAngles = MLX.cos(angles)
    let sinAngles = MLX.sin(angles)

    return MLX.stacked([cosAngles, sinAngles], axis: -1)
  }

  public func callAsFunction(
    videoFHW: [[(Int, Int, Int)]],
    txtSeqLens: [Int]
  ) -> (vid: MLXArray, txt: MLXArray) {
    let fhwList = videoFHW[0]  // Take first batch's shapes

    var vidFreqsList: [MLXArray] = []
    var maxVidIndex = 0
    let layerNum = fhwList.count - 1

    for (idx, fhw) in fhwList.enumerated() {
      let (frame, height, width) = fhw
      let videoFreq: MLXArray
      if idx != layerNum {
        // Regular layer
        videoFreq = computeVideoFreqs(frame: frame, height: height, width: width, idx: idx)
      } else {
        // Condition image uses negative index
        videoFreq = computeConditionFreqs(frame: frame, height: height, width: width)
      }
      vidFreqsList.append(videoFreq)

      if scaleRope {
        maxVidIndex = max(height / 2, width / 2, maxVidIndex)
      } else {
        maxVidIndex = max(height, width, maxVidIndex)
      }
    }

    maxVidIndex = max(maxVidIndex, layerNum)
    let maxLen = txtSeqLens.max() ?? 0

    // Text frequencies start after max video index
    let txtFreqs = posFreqs[maxVidIndex..<(maxVidIndex + maxLen), 0...]

    // Concatenate all video frequencies
    let vidFreqs = MLX.concatenated(vidFreqsList, axis: 0)

    return (vidFreqs, txtFreqs)
  }

  private func computeVideoFreqs(frame: Int, height: Int, width: Int, idx: Int) -> MLXArray {
    let seqLen = frame * height * width

    // Split frequencies by axis dimensions
    let halfDims = axesDim.map { $0 / 2 }
    let posFreq0 = posFreqs[0..., 0..<halfDims[0], 0...]
    let posFreq1 = posFreqs[0..., halfDims[0]..<(halfDims[0] + halfDims[1]), 0...]
    let posFreq2 = posFreqs[0..., (halfDims[0] + halfDims[1])..., 0...]

    // Frame frequencies (layer index)
    let freqsFrameRaw = posFreq0[idx..<(idx + frame), 0..., 0...]
      .reshaped([frame, 1, 1, halfDims[0], 2])
    let freqsFrame = MLX.broadcast(freqsFrameRaw, to: [frame, height, width, halfDims[0], 2])

    // Height and width frequencies
    let freqsHeight: MLXArray
    let freqsWidth: MLXArray

    if scaleRope {
      // Scale RoPE: use both positive and negative indices centered around 0
      let negFreq1 = negFreqs[0..., halfDims[0]..<(halfDims[0] + halfDims[1]), 0...]
      let negFreq2 = negFreqs[0..., (halfDims[0] + halfDims[1])..., 0...]

      let heightStart = height - height / 2
      let widthStart = width - width / 2

      let negHeightFreqs = negFreq1[(4096 - heightStart)..., 0..., 0...]
      let posHeightFreqs = posFreq1[0..<(height / 2), 0..., 0...]
      let heightFreqsCat = MLX.concatenated([negHeightFreqs, posHeightFreqs], axis: 0)
      let freqsHeightRaw = heightFreqsCat.reshaped([1, height, 1, halfDims[1], 2])
      freqsHeight = MLX.broadcast(freqsHeightRaw, to: [frame, height, width, halfDims[1], 2])

      let negWidthFreqs = negFreq2[(4096 - widthStart)..., 0..., 0...]
      let posWidthFreqs = posFreq2[0..<(width / 2), 0..., 0...]
      let widthFreqsCat = MLX.concatenated([negWidthFreqs, posWidthFreqs], axis: 0)
      let freqsWidthRaw = widthFreqsCat.reshaped([1, 1, width, halfDims[2], 2])
      freqsWidth = MLX.broadcast(freqsWidthRaw, to: [frame, height, width, halfDims[2], 2])
    } else {
      let freqsHeightRaw = posFreq1[0..<height, 0..., 0...]
        .reshaped([1, height, 1, halfDims[1], 2])
      freqsHeight = MLX.broadcast(freqsHeightRaw, to: [frame, height, width, halfDims[1], 2])

      let freqsWidthRaw = posFreq2[0..<width, 0..., 0...]
        .reshaped([1, 1, width, halfDims[2], 2])
      freqsWidth = MLX.broadcast(freqsWidthRaw, to: [frame, height, width, halfDims[2], 2])
    }

    // Concatenate all frequencies: [frame, height, width, total_dim/2, 2]
    let freqs = MLX.concatenated([freqsFrame, freqsHeight, freqsWidth], axis: -2)

    // Reshape to [seq_len, total_dim/2, 2]
    let totalHalfDim = halfDims.reduce(0, +)
    return freqs.reshaped([seqLen, totalHalfDim, 2])
  }

  private func computeConditionFreqs(frame: Int, height: Int, width: Int) -> MLXArray {
    let seqLen = frame * height * width

    let halfDims = axesDim.map { $0 / 2 }
    let negFreq0 = negFreqs[0..., 0..<halfDims[0], 0...]
    let posFreq1 = posFreqs[0..., halfDims[0]..<(halfDims[0] + halfDims[1]), 0...]
    let posFreq2 = posFreqs[0..., (halfDims[0] + halfDims[1])..., 0...]

    // Frame frequency: use last negative index (-1)
    let freqsFrameRaw = negFreq0[(4096 - 1)..., 0..., 0...]
      .reshaped([frame, 1, 1, halfDims[0], 2])
    let freqsFrame = MLX.broadcast(freqsFrameRaw, to: [frame, height, width, halfDims[0], 2])

    let freqsHeight: MLXArray
    let freqsWidth: MLXArray

    if scaleRope {
      let negFreq1 = negFreqs[0..., halfDims[0]..<(halfDims[0] + halfDims[1]), 0...]
      let negFreq2 = negFreqs[0..., (halfDims[0] + halfDims[1])..., 0...]

      let heightStart = height - height / 2
      let widthStart = width - width / 2

      let negHeightFreqs = negFreq1[(4096 - heightStart)..., 0..., 0...]
      let posHeightFreqs = posFreq1[0..<(height / 2), 0..., 0...]
      let heightFreqsCat = MLX.concatenated([negHeightFreqs, posHeightFreqs], axis: 0)
      let freqsHeightRaw = heightFreqsCat.reshaped([1, height, 1, halfDims[1], 2])
      freqsHeight = MLX.broadcast(freqsHeightRaw, to: [frame, height, width, halfDims[1], 2])

      let negWidthFreqs = negFreq2[(4096 - widthStart)..., 0..., 0...]
      let posWidthFreqs = posFreq2[0..<(width / 2), 0..., 0...]
      let widthFreqsCat = MLX.concatenated([negWidthFreqs, posWidthFreqs], axis: 0)
      let freqsWidthRaw = widthFreqsCat.reshaped([1, 1, width, halfDims[2], 2])
      freqsWidth = MLX.broadcast(freqsWidthRaw, to: [frame, height, width, halfDims[2], 2])
    } else {
      let freqsHeightRaw = posFreq1[0..<height, 0..., 0...]
        .reshaped([1, height, 1, halfDims[1], 2])
      freqsHeight = MLX.broadcast(freqsHeightRaw, to: [frame, height, width, halfDims[1], 2])

      let freqsWidthRaw = posFreq2[0..<width, 0..., 0...]
        .reshaped([1, 1, width, halfDims[2], 2])
      freqsWidth = MLX.broadcast(freqsWidthRaw, to: [frame, height, width, halfDims[2], 2])
    }

    let freqs = MLX.concatenated([freqsFrame, freqsHeight, freqsWidth], axis: -2)
    let totalHalfDim = halfDims.reduce(0, +)
    return freqs.reshaped([seqLen, totalHalfDim, 2])
  }
}

public func applyRotaryEmbQwen(_ x: MLXArray, freqs: MLXArray) -> MLXArray {
  // freqs: [S, D/2, 2] where last dim is (cos, sin)
  let cos = freqs[0..., 0..., 0]  // [S, D/2]
  let sin = freqs[0..., 0..., 1]  // [S, D/2]

  // Expand for batch and heads: [1, S, 1, D/2]
  let cosExp = cos[.newAxis, 0..., .newAxis, 0...]
  let sinExp = sin[.newAxis, 0..., .newAxis, 0...]

  // Split x into real and imaginary parts (interleaved)
  // x: [B, S, H, D] -> [B, S, H, D/2, 2]
  let xReshaped = x.reshaped(x.shape.dropLast() + [x.dim(-1) / 2, 2])
  let xReal = xReshaped[0..., 0..., 0..., 0..., 0]  // [B, S, H, D/2]
  let xImag = xReshaped[0..., 0..., 0..., 0..., 1]  // [B, S, H, D/2]

  // Apply rotation: (x_real + i*x_imag) * (cos + i*sin)
  // = x_real*cos - x_imag*sin + i*(x_real*sin + x_imag*cos)
  let outReal = xReal * cosExp - xImag * sinExp
  let outImag = xReal * sinExp + xImag * cosExp

  // Stack and reshape back to [B, S, H, D]
  let outStacked = MLX.stacked([outReal, outImag], axis: -1)
  return outStacked.reshaped(x.shape)
}
