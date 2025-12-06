import 'dart:math';
import 'package:kamma/kamma.dart';

/// Multi-head self-attention with cosine QK normalization for S3-DiT
class S3DiTAttention extends Module implements SimpleModule {
  final int hiddenSize;
  final int numHeads;
  final int headDim;
  final bool useQkNorm;
  final bool use3dRope;

  final LinearLayer qProj;
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer outProj;

  final LayerNorm? qNorm;
  final LayerNorm? kNorm;

  final Dropout attnDropout;
  final RoPE3D? rope;

  S3DiTAttention({
    required super.name,
    required this.hiddenSize,
    required this.numHeads,
    required this.headDim,
    required this.useQkNorm,
    required this.use3dRope,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.outProj,
    this.qNorm,
    this.kNorm,
    required this.attnDropout,
    this.rope,
  });

  /// Split tensor into multiple heads
  /// Input: [batch, seqLen, hiddenSize]
  /// Output: [batch, seqLen, numHeads, headDim]
  Tensor _splitHeads(Tensor x) {
    final batch = x.shape[0];
    final seqLen = x.shape[1];

    return x.view([batch, seqLen, numHeads, headDim]);
  }

  /// Merge multiple heads back
  /// Input: [batch, seqLen, numHeads, headDim]
  /// Output: [batch, seqLen, hiddenSize]
  Tensor _mergeHeads(Tensor x) {
    final batch = x.shape[0];
    final seqLen = x.shape[1];

    return x.view([batch, seqLen, hiddenSize]);
  }

  /// Apply cosine normalization to query and key
  Tensor _cosineNorm(Tensor x) {
    // Normalize along the head dimension
    final Tensor norm = x.norm(2, dim: [-1], keepDim: true);
    return x / (norm + 1e-6);
  }

  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? posX,
    Tensor? posY,
    Tensor? posT,
    Tensor? attentionMask,
    required Context context,
  }) {
    context.onloadModule(this);

    final batch = hiddenStates.shape[0];
    final seqLen = hiddenStates.shape[1];

    // Project to Q, K, V
    Tensor query = qProj.forward(hiddenStates, context: context);
    Tensor key = kProj.forward(hiddenStates, context: context);
    Tensor value = vProj.forward(hiddenStates, context: context);

    // Split into heads
    query = _splitHeads(query);
    key = _splitHeads(key);
    value = _splitHeads(value);

    // Apply QK normalization if enabled
    if (useQkNorm && qNorm != null && kNorm != null) {
      // Reshape for layer norm: [batch * seqLen * numHeads, headDim]
      final qReshaped = query.view([batch * seqLen * numHeads, headDim]);
      final kReshaped = key.view([batch * seqLen * numHeads, headDim]);

      query = qNorm!.forward(qReshaped, context: context);
      key = kNorm!.forward(kReshaped, context: context);

      // Reshape back: [batch, seqLen, numHeads, headDim]
      query = query.view([batch, seqLen, numHeads, headDim]);
      key = key.view([batch, seqLen, numHeads, headDim]);

      // Apply cosine normalization
      query = _cosineNorm(query);
      key = _cosineNorm(key);
    }

    // Apply 3D RoPE if enabled
    if (use3dRope && rope != null) {
      if (posX == null || posY == null || posT == null) {
        throw ArgumentError(
          '3D RoPE is enabled but position indices (posX, posY, posT) are not provided',
        );
      }

      query = rope!.apply(query, posX: posX, posY: posY, posT: posT);
      key = rope!.apply(key, posX: posX, posY: posY, posT: posT);
    }

    // Transpose for attention computation
    // From [batch, seqLen, numHeads, headDim]
    // To [batch, numHeads, seqLen, headDim]
    query = query.permute([0, 2, 1, 3]);
    key = key.permute([0, 2, 1, 3]);
    value = value.permute([0, 2, 1, 3]);

    // Compute attention scores
    // [batch, numHeads, seqLen, seqLen]
    Tensor attnWeights = query.matmul(key.transpose(-2, -1));

    // Scale by sqrt(headDim)
    attnWeights = attnWeights / sqrt(headDim.toDouble());

    // Apply attention mask if provided
    if (attentionMask != null) {
      attnWeights = attnWeights + attentionMask;
    }

    // Softmax
    attnWeights = attnWeights.softmax(-1);

    // Apply dropout
    attnWeights = attnDropout.forward(attnWeights, context: context);

    // Apply attention to values
    // [batch, numHeads, seqLen, headDim]
    Tensor attnOutput = attnWeights.matmul(value);

    // Transpose back to [batch, seqLen, numHeads, headDim]
    attnOutput = attnOutput.permute([0, 2, 1, 3]);

    // Merge heads
    attnOutput = _mergeHeads(attnOutput);

    // Output projection
    attnOutput = outProj.forward(attnOutput, context: context);

    return attnOutput;
  }

  @override
  void resetParameters() {
    qProj.resetParameters();
    kProj.resetParameters();
    vProj.resetParameters();
    outProj.resetParameters();
    qNorm?.resetParameters();
    kNorm?.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...qProj.parameters,
    ...kProj.parameters,
    ...vProj.parameters,
    ...outProj.parameters,
    if (qNorm != null) ...qNorm!.parameters,
    if (kNorm != null) ...kNorm!.parameters,
  ];

  @override
  late final Iterable<Module> submodules = [
    qProj,
    kProj,
    vProj,
    outProj,
    if (qNorm != null) qNorm!,
    if (kNorm != null) kNorm!,
    attnDropout,
    // Note: rope is not a Module, so it's not included in submodules
  ];

  @override
  Map<String, dynamic> get meta => {
    'hiddenSize': hiddenSize,
    'numHeads': numHeads,
    'headDim': headDim,
    'useQkNorm': useQkNorm,
    'use3dRope': use3dRope,
  };

  static S3DiTAttention make({
    required S3DiTConfig config,
    required String name,
  }) {
    final qProj = LinearLayer.make(
      name: 'q_proj',
      inFeatures: config.hiddenSize,
      outFeatures: config.hiddenSize,
    );

    final kProj = LinearLayer.make(
      name: 'k_proj',
      inFeatures: config.hiddenSize,
      outFeatures: config.hiddenSize,
    );

    final vProj = LinearLayer.make(
      name: 'v_proj',
      inFeatures: config.hiddenSize,
      outFeatures: config.hiddenSize,
    );

    final outProj = LinearLayer.make(
      name: 'out_proj',
      inFeatures: config.hiddenSize,
      outFeatures: config.hiddenSize,
    );

    LayerNorm? qNorm;
    LayerNorm? kNorm;
    if (config.useQkNorm) {
      qNorm = LayerNorm.make(
        name: 'q_norm',
        normalizedShape: [config.headDim],
        eps: config.layerNormEps,
      );
      kNorm = LayerNorm.make(
        name: 'k_norm',
        normalizedShape: [config.headDim],
        eps: config.layerNormEps,
      );
    }

    final attnDropout = Dropout(config.attentionDropout);

    RoPE3D? rope;
    if (config.use3dRope) {
      rope = RoPE3D(
        dim: config.headDim,
        theta: config.ropeTheta,
        maxSeqLen: config.maxPositionEmbeddings,
      );
    }

    return S3DiTAttention(
      name: name,
      hiddenSize: config.hiddenSize,
      numHeads: config.numAttentionHeads,
      headDim: config.headDim,
      useQkNorm: config.useQkNorm,
      use3dRope: config.use3dRope,
      qProj: qProj,
      kProj: kProj,
      vProj: vProj,
      outProj: outProj,
      qNorm: qNorm,
      kNorm: kNorm,
      attnDropout: attnDropout,
      rope: rope,
    );
  }
}
