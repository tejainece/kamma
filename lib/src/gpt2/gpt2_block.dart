import 'package:kamma/kamma.dart';

class GPT2Block extends Module implements SimpleModule {
  final LayerNorm ln1;
  final GPT2Attention attn;
  final LayerNorm ln2;
  final GPT2MLP mlp;

  GPT2Block({
    required super.name,
    required this.ln1,
    required this.attn,
    required this.ln2,
    required this.mlp,
  });

  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? layerPast,
    Tensor? attentionMask,
    Tensor? headMask,
    Tensor? encoderHiddenStates,
    bool outputAttentions = false,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor residual = hiddenStates;
    hiddenStates = ln1.forward(hiddenStates, context: context);

    // TODO setup keyValueCache
    // TODO use attentionWeights
    final (:outputEmbeddings, :attentionWeights) = attn.forward(
      hiddenStates,
      attentionMask: attentionMask,
      headMask: headMask,
      encoderHiddenStates: encoderHiddenStates,
      outputAttentions: outputAttentions,
      context: context,
    );

    // TODO: Handle attnOutput being a tuple if useCache or outputAttentions is true
    // For now assuming it returns just the attention output tensor

    hiddenStates = outputEmbeddings + residual;

    residual = hiddenStates;
    hiddenStates = ln2.forward(hiddenStates, context: context);
    hiddenStates = mlp.forward(hiddenStates, context: context);
    hiddenStates = hiddenStates + residual;

    return hiddenStates;
  }

  @override
  void resetParameters() {
    ln1.resetParameters();
    attn.resetParameters();
    ln2.resetParameters();
    mlp.resetParameters();
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  Map<String, dynamic> get meta => {};

  @override
  late final Iterable<Module> submodules = [ln1, attn, ln2, mlp];

  static GPT2Block make({
    required String name,
    required int embedDim,
    required int numHeads,
    required double layerNormEpsilon,
    int layerIdx = 0,
    String attentionName = 'attn',
    String preLayerNormName = 'ln_1',
    String postLayerNormName = 'ln_2',
    String mlpName = 'mlp',
    required bool scaleAttnByInverseLayerIdx,
    required int mlpInnerDim,
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required bool isCrossAttention,
    required int maxPositionEmbeddings,
  }) {
    final ln1 = LayerNorm.make(
      name: 'ln_1',
      normalizedShape: [embedDim],
      eps: layerNormEpsilon,
    );

    final attn = GPT2Attention.make(
      name: 'attn',
      layerIdx: layerIdx,
      embedDim: embedDim,
      numHeads: numHeads,
      attentionDropoutProbability: attentionDropoutProbability,
      residualDropoutProbability: residualDropoutProbability,
      isCrossAttention: isCrossAttention,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );

    final ln2 = LayerNorm.make(
      name: 'ln_2',
      normalizedShape: [embedDim],
      eps: layerNormEpsilon,
    );

    final mlp = GPT2MLP.make(
      name: 'mlp',
      embedDim: embedDim,
      mlpInnerDim: mlpInnerDim,
      residualDropoutProbability: residualDropoutProbability,
    );

    return GPT2Block(name: name, ln1: ln1, attn: attn, ln2: ln2, mlp: mlp);
  }

  static Future<GPT2Block> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required double layerNormEpsilon,
    String attentionName = 'attn',
    String preLayerNormName = 'ln_1',
    String postLayerNormName = 'ln_2',
    String mlpName = 'mlp',
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required int numHeads,
    required bool isCrossAttention,
    required int layerIdx,
    required bool scaleAttnByInverseLayerIdx,
    required int maxPositionEmbeddings,
  }) async {
    final attn = await GPT2Attention.loadFromSafeTensor(
      loader,
      prefix: '$prefix$attentionName.',
      name: attentionName,
      attentionDropoutProbability: attentionDropoutProbability,
      residualDropoutProbability: residualDropoutProbability,
      isCrossAttention: isCrossAttention,
      numHeads: numHeads,
      layerIdx: layerIdx,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );
    final embedDim = attn.embedDim;
    final ln1 = await LayerNorm.loadFromSafeTensor(
      loader,
      name: preLayerNormName,
      prefix: '$prefix$preLayerNormName.',
      normalizedShape: [embedDim],
      eps: layerNormEpsilon,
    );
    final ln2 = await LayerNorm.loadFromSafeTensor(
      loader,
      name: postLayerNormName,
      prefix: '$prefix$postLayerNormName.',
      normalizedShape: [embedDim],
      eps: layerNormEpsilon,
    );
    final mlp = await GPT2MLP.loadFromSafeTensor(
      loader,
      prefix: '$prefix$mlpName.',
      name: mlpName,
      residualDropoutProbability: residualDropoutProbability,
    );
    return GPT2Block(name: name, ln1: ln1, attn: attn, ln2: ln2, mlp: mlp);
  }
}
