import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Layered Configuration

/// Configuration for the Qwen-Image-Layered transformer
public struct QwenLayeredTransformerConfiguration: Codable, Sendable {
  public var numLayers: Int
  public var numAttentionHeads: Int
  public var attentionHeadDim: Int
  public var inChannels: Int
  public var outChannels: Int
  public var jointAttentionDim: Int
  public var patchSize: Int
  public var axesDimsRope: [Int]
  public var guidanceEmbeds: Bool
  public var zeroCondT: Bool
  public var useAdditionalTCond: Bool
  public var useLayer3dRope: Bool

  public var innerDim: Int {
    numAttentionHeads * attentionHeadDim
  }

  enum CodingKeys: String, CodingKey {
    case numLayers = "num_layers"
    case numAttentionHeads = "num_attention_heads"
    case attentionHeadDim = "attention_head_dim"
    case inChannels = "in_channels"
    case outChannels = "out_channels"
    case jointAttentionDim = "joint_attention_dim"
    case patchSize = "patch_size"
    case axesDimsRope = "axes_dims_rope"
    case guidanceEmbeds = "guidance_embeds"
    case zeroCondT = "zero_cond_t"
    case useAdditionalTCond = "use_additional_t_cond"
    case useLayer3dRope = "use_layer3d_rope"
  }

  public init(
    numLayers: Int = 60,
    numAttentionHeads: Int = 24,
    attentionHeadDim: Int = 128,
    inChannels: Int = 64,
    outChannels: Int = 16,
    jointAttentionDim: Int = 3584,
    patchSize: Int = 2,
    axesDimsRope: [Int] = [16, 56, 56],
    guidanceEmbeds: Bool = false,
    zeroCondT: Bool = true,
    useAdditionalTCond: Bool = true,
    useLayer3dRope: Bool = true
  ) {
    self.numLayers = numLayers
    self.numAttentionHeads = numAttentionHeads
    self.attentionHeadDim = attentionHeadDim
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.jointAttentionDim = jointAttentionDim
    self.patchSize = patchSize
    self.axesDimsRope = axesDimsRope
    self.guidanceEmbeds = guidanceEmbeds
    self.zeroCondT = zeroCondT
    self.useAdditionalTCond = useAdditionalTCond
    self.useLayer3dRope = useLayer3dRope
  }

  public static func load(from directory: URL) throws -> QwenLayeredTransformerConfiguration {
    let configURL = directory.appending(path: "config.json")
    guard FileManager.default.fileExists(atPath: configURL.path) else {
      return QwenLayeredTransformerConfiguration()
    }
    let data = try Data(contentsOf: configURL)
    let decoder = JSONDecoder()
    return try decoder.decode(QwenLayeredTransformerConfiguration.self, from: data)
  }
}

// MARK: - Modulation MLP

public final class LayeredModulationMLP: Module {
  @ModuleInfo(key: "lin") var linear: Linear

  public init(inputDim: Int, outputDim: Int) {
    self._linear.wrappedValue = Linear(inputDim, outputDim, bias: true)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    return linear(MLXNN.silu(x))
  }
}

// MARK: - Feed-forward Network

public final class LayeredFeedForward: Module {
  @ModuleInfo(key: "linear1") var linear1: Linear
  @ModuleInfo(key: "linear2") var linear2: Linear

  public init(dim: Int, hiddenDim: Int? = nil) {
    let hidden = hiddenDim ?? dim * 4
    self._linear1.wrappedValue = Linear(dim, hidden, bias: true)
    self._linear2.wrappedValue = Linear(hidden, dim, bias: true)
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var x = linear1(x)
    x = MLXNN.geluApproximate(x)
    x = linear2(x)
    return x
  }
}

// MARK: - Timestep Embeddings

public final class LayeredTimesteps: Module {
  let numChannels: Int
  let flipSinToCos: Bool
  let downscaleFreqShift: Float
  let scale: Float

  public init(
    numChannels: Int = 256,
    flipSinToCos: Bool = true,
    downscaleFreqShift: Float = 0,
    scale: Float = 1000
  ) {
    self.numChannels = numChannels
    self.flipSinToCos = flipSinToCos
    self.downscaleFreqShift = downscaleFreqShift
    self.scale = scale
  }

  public func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
    let halfDim = numChannels / 2
    let maxPeriod: Float = 10000

    let indices = MLXArray(stride(from: 0, to: halfDim, by: 1).map { Float($0) })
    let exponent = -log(maxPeriod) * indices / (Float(halfDim) - downscaleFreqShift)
    var emb = MLX.exp(exponent)

    emb = timesteps[0..., .newAxis].asType(.float32) * emb[.newAxis, 0...]
    emb = scale * emb

    let sinEmb = MLX.sin(emb)
    let cosEmb = MLX.cos(emb)

    if flipSinToCos {
      emb = MLX.concatenated([cosEmb, sinEmb], axis: -1)
    } else {
      emb = MLX.concatenated([sinEmb, cosEmb], axis: -1)
    }

    return emb
  }
}

public final class LayeredTimestepEmbedding: Module {
  @ModuleInfo(key: "linear_1") public var linear1: Linear
  @ModuleInfo(key: "linear_2") public var linear2: Linear

  public init(inChannels: Int, timeEmbedDim: Int) {
    self._linear1.wrappedValue = Linear(inChannels, timeEmbedDim, bias: true)
    self._linear2.wrappedValue = Linear(timeEmbedDim, timeEmbedDim, bias: true)
  }

  public func callAsFunction(_ sample: MLXArray) -> MLXArray {
    var x = linear1(sample)
    x = MLXNN.silu(x)
    x = linear2(x)
    return x
  }
}

public final class LayeredTimestepProjEmbeddings: Module {
  @ModuleInfo(key: "time_proj") public var timeProj: LayeredTimesteps
  @ModuleInfo(key: "timestep_embedder") public var timestepEmbedder: LayeredTimestepEmbedding
  @ModuleInfo(key: "addition_t_embedding") public var additionTEmbedding: Embedding?

  let useAdditionalTCond: Bool

  public init(embeddingDim: Int, useAdditionalTCond: Bool = false) {
    self.useAdditionalTCond = useAdditionalTCond

    self._timeProj.wrappedValue = LayeredTimesteps(
      numChannels: 256,
      flipSinToCos: true,
      downscaleFreqShift: 0,
      scale: 1000
    )
    self._timestepEmbedder.wrappedValue = LayeredTimestepEmbedding(
      inChannels: 256,
      timeEmbedDim: embeddingDim
    )

    if useAdditionalTCond {
      self._additionTEmbedding.wrappedValue = Embedding(embeddingCount: 2, dimensions: embeddingDim)
    } else {
      self._additionTEmbedding.wrappedValue = nil
    }
  }

  public func callAsFunction(
    timestep: MLXArray,
    hiddenStates: MLXArray,
    additionalTCond: MLXArray? = nil
  ) -> MLXArray {
    let timestepsProj = timeProj(timestep)
    var timestepsEmb = timestepEmbedder(timestepsProj.asType(hiddenStates.dtype))

    if useAdditionalTCond, let additionTCond = additionalTCond, let embedding = additionTEmbedding
    {
      let additionTEmb = embedding(additionTCond).asType(hiddenStates.dtype)
      timestepsEmb = timestepsEmb + additionTEmb
    }

    return timestepsEmb
  }
}

// MARK: - Adaptive Layer Norm

public final class LayeredAdaLayerNormContinuous: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  @ModuleInfo(key: "norm") var norm: LayerNorm

  let embeddingDim: Int

  public init(embeddingDim: Int, conditioningEmbeddingDim: Int) {
    self.embeddingDim = embeddingDim
    self._linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2, bias: true)
    self._norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false)
  }

  public func callAsFunction(_ x: MLXArray, conditioning: MLXArray) -> MLXArray {
    let emb = linear(MLXNN.silu(conditioning))
    let chunks = emb.split(parts: 2, axis: -1)
    let scale = chunks[0]
    let shift = chunks[1]
    let normalized = norm(x)
    return normalized * (1 + scale[0..., .newAxis, 0...]) + shift[0..., .newAxis, 0...]
  }
}

// MARK: - Layered Transformer Block

public final class QwenLayeredTransformerBlockV2: Module {
  let dim: Int
  let numHeads: Int
  let headDim: Int
  let zeroCondT: Bool

  @ModuleInfo(key: "img_mod") var imgMod: LayeredModulationMLP
  @ModuleInfo(key: "img_norm1") var imgNorm1: LayerNorm
  @ModuleInfo(key: "img_norm2") var imgNorm2: LayerNorm
  @ModuleInfo(key: "img_mlp") var imgMlp: LayeredFeedForward

  @ModuleInfo(key: "txt_mod") var txtMod: LayeredModulationMLP
  @ModuleInfo(key: "txt_norm1") var txtNorm1: LayerNorm
  @ModuleInfo(key: "txt_norm2") var txtNorm2: LayerNorm
  @ModuleInfo(key: "txt_mlp") var txtMlp: LayeredFeedForward

  @ModuleInfo(key: "attn") public var attn: QwenLayeredJointAttentionV2

  public init(dim: Int, numHeads: Int, headDim: Int, zeroCondT: Bool = false) {
    self.dim = dim
    self.numHeads = numHeads
    self.headDim = headDim
    self.zeroCondT = zeroCondT

    self._imgMod.wrappedValue = LayeredModulationMLP(inputDim: dim, outputDim: 6 * dim)
    self._imgNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._imgNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._imgMlp.wrappedValue = LayeredFeedForward(dim: dim)

    self._txtMod.wrappedValue = LayeredModulationMLP(inputDim: dim, outputDim: 6 * dim)
    self._txtNorm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._txtNorm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    self._txtMlp.wrappedValue = LayeredFeedForward(dim: dim)

    self._attn.wrappedValue = QwenLayeredJointAttentionV2(dim: dim, numHeads: numHeads, headDim: headDim)
  }

  private func modulate(
    _ x: MLXArray,
    modParams: MLXArray,
    index: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    let chunks = modParams.split(parts: 3, axis: -1)
    var shift = chunks[0]
    var scale = chunks[1]
    var gate = chunks[2]

    if let index = index {
      let actualBatch = shift.dim(0) / 2
      let shift0 = shift[0..<actualBatch, 0...]
      let shift1 = shift[actualBatch..., 0...]
      let scale0 = scale[0..<actualBatch, 0...]
      let scale1 = scale[actualBatch..., 0...]
      let gate0 = gate[0..<actualBatch, 0...]
      let gate1 = gate[actualBatch..., 0...]

      let indexExp = index[0..., 0..., .newAxis]

      shift = MLX.where(indexExp .== 0, shift0[0..., .newAxis, 0...], shift1[0..., .newAxis, 0...])
      scale = MLX.where(indexExp .== 0, scale0[0..., .newAxis, 0...], scale1[0..., .newAxis, 0...])
      gate = MLX.where(indexExp .== 0, gate0[0..., .newAxis, 0...], gate1[0..., .newAxis, 0...])
    } else {
      shift = shift[0..., .newAxis, 0...]
      scale = scale[0..., .newAxis, 0...]
      gate = gate[0..., .newAxis, 0...]
    }

    let modulated = x * (1 + scale) + shift
    return (modulated, gate)
  }

  public func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    temb: MLXArray,
    imageRotaryEmb: (vid: MLXArray, txt: MLXArray),
    modulateIndex: MLXArray? = nil
  ) -> (MLXArray, MLXArray) {
    let imgModParams = imgMod(temb)

    var txtTemb = temb
    if zeroCondT {
      txtTemb = temb.split(parts: 2, axis: 0)[0]
    }
    let txtModParams = txtMod(txtTemb)

    let imgMod1 = imgModParams[0..., 0..<(3 * dim)]
    let imgMod2 = imgModParams[0..., (3 * dim)...]
    let txtMod1 = txtModParams[0..., 0..<(3 * dim)]
    let txtMod2 = txtModParams[0..., (3 * dim)...]

    let imgNormed = imgNorm1(hiddenStates)
    let (imgModulated, imgGate1) = modulate(imgNormed, modParams: imgMod1, index: modulateIndex)

    let txtNormed = txtNorm1(encoderHiddenStates)
    let (txtModulated, txtGate1) = modulate(txtNormed, modParams: txtMod1)

    let (imgAttnOutput, txtAttnOutput) = attn(
      hiddenStates: imgModulated,
      encoderHiddenStates: txtModulated,
      imageRotaryEmb: imageRotaryEmb
    )

    var newHiddenStates = hiddenStates + imgGate1 * imgAttnOutput
    var newEncoderHiddenStates = encoderHiddenStates + txtGate1 * txtAttnOutput

    let imgNormed2 = imgNorm2(newHiddenStates)
    let (imgModulated2, imgGate2) = modulate(imgNormed2, modParams: imgMod2, index: modulateIndex)
    let imgMlpOutput = imgMlp(imgModulated2)
    newHiddenStates = newHiddenStates + imgGate2 * imgMlpOutput

    let txtNormed2 = txtNorm2(newEncoderHiddenStates)
    let (txtModulated2, txtGate2) = modulate(txtNormed2, modParams: txtMod2)
    let txtMlpOutput = txtMlp(txtModulated2)
    newEncoderHiddenStates = newEncoderHiddenStates + txtGate2 * txtMlpOutput

    if newEncoderHiddenStates.dtype == .float16 {
      newEncoderHiddenStates = MLX.clip(newEncoderHiddenStates, min: -65504, max: 65504)
    }
    if newHiddenStates.dtype == .float16 {
      newHiddenStates = MLX.clip(newHiddenStates, min: -65504, max: 65504)
    }

    return (newEncoderHiddenStates, newHiddenStates)
  }
}

// MARK: - Full Transformer

public final class QwenLayeredTransformerV2: Module {
  public let configuration: QwenLayeredTransformerConfiguration

  @ModuleInfo(key: "img_in") public var imgIn: Linear
  @ModuleInfo(key: "txt_in") public var txtIn: Linear
  @ModuleInfo(key: "txt_norm") public var txtNorm: RMSNorm

  @ModuleInfo(key: "time_text_embed") public var timeTextEmbed: LayeredTimestepProjEmbeddings
  @ModuleInfo(key: "pos_embed") public var posEmbed: QwenLayeredRoPE

  @ModuleInfo(key: "transformer_blocks") public var transformerBlocks: [QwenLayeredTransformerBlockV2]

  @ModuleInfo(key: "norm_out") public var normOut: LayeredAdaLayerNormContinuous
  @ModuleInfo(key: "proj_out") public var projOut: Linear

  public init(_ configuration: QwenLayeredTransformerConfiguration) {
    self.configuration = configuration

    let innerDim = configuration.innerDim

    self._imgIn.wrappedValue = Linear(configuration.inChannels, innerDim, bias: true)
    self._txtIn.wrappedValue = Linear(configuration.jointAttentionDim, innerDim, bias: true)
    self._txtNorm.wrappedValue = RMSNorm(dimensions: configuration.jointAttentionDim, eps: 1e-6)

    self._timeTextEmbed.wrappedValue = LayeredTimestepProjEmbeddings(
      embeddingDim: innerDim,
      useAdditionalTCond: configuration.useAdditionalTCond
    )
    self._posEmbed.wrappedValue = QwenLayeredRoPE(
      theta: 10000,
      axesDim: configuration.axesDimsRope,
      scaleRope: true
    )

    self._transformerBlocks.wrappedValue = (0..<configuration.numLayers).map { _ in
      QwenLayeredTransformerBlockV2(
        dim: innerDim,
        numHeads: configuration.numAttentionHeads,
        headDim: configuration.attentionHeadDim,
        zeroCondT: configuration.zeroCondT
      )
    }

    self._normOut.wrappedValue = LayeredAdaLayerNormContinuous(
      embeddingDim: innerDim,
      conditioningEmbeddingDim: innerDim
    )
    let outDim = configuration.patchSize * configuration.patchSize * configuration.outChannels
    self._projOut.wrappedValue = Linear(innerDim, outDim, bias: true)
  }

  public func callAsFunction(
    hiddenStates: MLXArray,
    encoderHiddenStates: MLXArray,
    encoderHiddenStatesMask: MLXArray?,
    timestep: MLXArray,
    imgShapes: [[(Int, Int, Int)]],
    txtSeqLens: [Int],
    additionalTCond: MLXArray? = nil
  ) -> MLXArray {
    var hiddenStatesMut = imgIn(hiddenStates)

    var timestep = timestep.asType(hiddenStatesMut.dtype)
    var modulateIndex: MLXArray? = nil

    if configuration.zeroCondT {
      let zeroTimestep = timestep * 0
      timestep = MLX.concatenated([timestep, zeroTimestep], axis: 0)

      var indexList: [[Int]] = []
      for shapes in imgShapes {
        let firstLayerSize = shapes[0].0 * shapes[0].1 * shapes[0].2
        var indices = [Int](repeating: 0, count: firstLayerSize)
        for i in 1..<shapes.count {
          let size = shapes[i].0 * shapes[i].1 * shapes[i].2
          indices.append(contentsOf: [Int](repeating: 1, count: size))
        }
        indexList.append(indices)
      }
      let maxLen = indexList.map { $0.count }.max() ?? 0
      let paddedIndices = indexList.map { arr in
        arr + [Int](repeating: 0, count: maxLen - arr.count)
      }
      modulateIndex = MLXArray(paddedIndices.flatMap { $0 }).reshaped([
        imgShapes.count, maxLen,
      ])
    }

    let encoderHiddenStatesNorm = txtNorm(encoderHiddenStates)
    var encoderHiddenStatesMut = txtIn(encoderHiddenStatesNorm)

    let temb = timeTextEmbed(
      timestep: timestep,
      hiddenStates: hiddenStatesMut,
      additionalTCond: additionalTCond
    )

    let imageRotaryEmb = posEmbed(videoFHW: imgShapes, txtSeqLens: txtSeqLens)

    for block in transformerBlocks {
      (encoderHiddenStatesMut, hiddenStatesMut) = block(
        hiddenStates: hiddenStatesMut,
        encoderHiddenStates: encoderHiddenStatesMut,
        temb: temb,
        imageRotaryEmb: imageRotaryEmb,
        modulateIndex: modulateIndex
      )
    }

    var finalTemb = temb
    if configuration.zeroCondT {
      finalTemb = temb.split(parts: 2, axis: 0)[0]
    }

    hiddenStatesMut = normOut(hiddenStatesMut, conditioning: finalTemb)
    let output = projOut(hiddenStatesMut)

    return output
  }
}
