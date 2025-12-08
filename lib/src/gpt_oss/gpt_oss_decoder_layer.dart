import 'package:kamma/kamma.dart';

/// TODO GradientCheckpointingLayer
class GptOssDecoderLayer extends Module implements SimpleModule {
  final RMSNorm ln1;
  final GptOssAttention attn;
  final RMSNorm ln2;
  final GptOssMoE moe;

  GptOssDecoderLayer({
    required super.name,
    required this.ln1,
    required this.attn,
    required this.ln2,
    required this.moe,
  });

  @override
  Tensor forward(
    Tensor hiddenStates, {
    List<Tensor>? layerPast,
    Tensor? attentionMask,
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? encoderHiddenStates,
    Tensor? encoderAttentionMask,
    bool useCache = false,
    bool outputAttentions = false,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor residual = hiddenStates;
    hiddenStates = ln1.forward(hiddenStates, context: context);

    Tensor attnOutput = attn.forward(
      hiddenStates,
      layerPast: layerPast,
      attentionMask: attentionMask,
      positionIds: positionIds,
      headMask: headMask,
      encoderHiddenStates: encoderHiddenStates,
      encoderAttentionMask: encoderAttentionMask,
      useCache: useCache,
      outputAttentions: outputAttentions,
      context: context,
    );

    hiddenStates = attnOutput + residual;

    residual = hiddenStates;
    hiddenStates = ln2.forward(hiddenStates, context: context);
    hiddenStates = moe.forward(hiddenStates, context: context);
    hiddenStates = hiddenStates + residual;

    return hiddenStates;
  }

  @override
  void resetParameters() {
    ln1.resetParameters();
    attn.resetParameters();
    ln2.resetParameters();
    moe.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...ln1.parameters,
    ...attn.parameters,
    ...ln2.parameters,
    ...moe.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {};

  @override
  late final Iterable<Module> submodules = [ln1, attn, ln2, moe];

  static GptOssDecoderLayer make({
    required String name,
    required int embedDim,
    required int numHeads,
    required int nInner,
    required int nPositions,
    required double attentionDropoutP,
    required double residDropoutP,
    required int numKeyValueHeads,
    required double ropeTheta,
    required bool isCrossAttention,
    required int numExperts,
    required int numExpertsPerToken,
    required double rmsNormEps,
    int layerIdx = 0,
  }) {
    final ln1 = RMSNorm.make(
      name: 'ln_1',
      normalizedShape: [embedDim],
      eps: rmsNormEps,
    );

    final attention = GptOssAttention.make(
      name: 'attn',
      layerIdx: layerIdx,
      embedDim: embedDim,
      numHeads: numHeads,
      nPositions: nPositions,
      attentionDropoutP: attentionDropoutP,
      residDropoutP: residDropoutP,
      numKeyValueHeads: numKeyValueHeads,
      ropeTheta: ropeTheta,
      isCrossAttention: isCrossAttention,
    );

    final ln2 = RMSNorm.make(
      name: 'ln_2',
      normalizedShape: [embedDim],
      eps: rmsNormEps,
    );

    final moe = GptOssMoE.make(
      name: 'moe',
      embedDim: embedDim,
      nInner: nInner,
      numExperts: numExperts,
      numExpertsPerToken: numExpertsPerToken,
    );

    return GptOssDecoderLayer(
      name: name,
      ln1: ln1,
      attn: attention,
      ln2: ln2,
      moe: moe,
    );
  }

  static Future<GptOssDecoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int embedDim,
    required int numHeads,
    required int nInner,
    required int nPositions,
    required double attentionDropoutP,
    required double residDropoutP,
    required int numKeyValueHeads,
    required double ropeTheta,
    required bool isCrossAttention,
    required int numExperts,
    required int numExpertsPerToken,
    required double rmsNormEps,
    String? preAttentionLayerNormName,
    String? postAttentionLayerNormName,
    String attentionName = 'self_attn',
    String moeName = 'mlp',
  }) async {
    // ln_1 is now RMSNorm, might be named 'input_layernorm' or 'ln_1' depending on mapping.
    // Assuming standard gpt-oss naming: input_layernorm
    RMSNorm ln1;
    if (preAttentionLayerNormName == null) {
      if (loader.hasTensor('${prefix}input_layernorm.weight')) {
        preAttentionLayerNormName = 'input_layernorm';
      } else {
        preAttentionLayerNormName = 'ln_1';
      }
    }
    ln1 = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix$preAttentionLayerNormName.',
      name: preAttentionLayerNormName,
      normalizedShape: [embedDim],
      eps: rmsNormEps,
    );

    GptOssAttention attention = await GptOssAttention.loadFromSafeTensor(
      loader,
      prefix: '$prefix$attentionName.',
      name: attentionName,
      embedDim: embedDim,
      numHeads: numHeads,
      nPositions: nPositions,
      attentionDropoutP: attentionDropoutP,
      residDropoutP: residDropoutP,
      numKeyValueHeads: numKeyValueHeads,
      ropeTheta: ropeTheta,
      isCrossAttention: isCrossAttention,
      layerIdx: 0,
    );

    RMSNorm ln2;
    if (postAttentionLayerNormName == null) {
      if (loader.hasTensor('${prefix}post_attention_layernorm.weight')) {
        postAttentionLayerNormName = 'post_attention_layernorm';
      } else {
        postAttentionLayerNormName = 'ln_2';
      }
    }
    ln2 = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix$postAttentionLayerNormName.',
      name: postAttentionLayerNormName,
      normalizedShape: [embedDim],
      eps: rmsNormEps,
    );

    GptOssMoE moe = await GptOssMoE.loadFromSafeTensor(
      loader,
      prefix: '$prefix$moeName.',
      name: moeName,
      numExpertsPerToken: numExpertsPerToken,
      numExperts: numExperts,
      embedDim: embedDim,
      nInner: nInner,
    );

    return GptOssDecoderLayer(
      name: name,
      ln1: ln1,
      attn: attention,
      ln2: ln2,
      moe: moe,
    );
  }
}
