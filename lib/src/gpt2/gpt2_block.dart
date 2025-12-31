import 'package:kamma/kamma.dart';

class GPT2Block extends Module implements SimpleModule {
  final LayerNorm ln1;
  final GPT2Attention attention;
  final LayerNorm ln2;

  final GPT2Attention? crossAttention;
  final LayerNorm? lnCrossAttention;

  final GPT2MLP mlp;

  GPT2Block({
    required super.name,
    required this.ln1,
    required this.attention,
    required this.ln2,
    required this.crossAttention,
    required this.lnCrossAttention,
    required this.mlp,
  });

  @override
  Tensor forward(
    Tensor embeddings, {
    Tensor? attentionMask,
    Tensor? headMask,
    Tensor? encoderHiddenStates,
    List<Tensor>? outputSelfAttentions,
    List<Tensor>? outputCrossAttention,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor residual = embeddings;

    embeddings = ln1.forward(embeddings, context: context);
    final outputEmbeddings = attention.forward(
      embeddings,
      attentionMask: attentionMask,
      // TODO cache position
      headMask: headMask,
      encoderHiddenStates: encoderHiddenStates,
      outputAttentions: outputSelfAttentions,
      context: context,
    );
    embeddings = outputEmbeddings + residual;

    if (crossAttention != null) {
      residual = embeddings;
      embeddings = lnCrossAttention!.forward(embeddings, context: context);
      Tensor crossAttentionOutput = crossAttention!.forward(
        embeddings,
        context: context,
        attentionMask: attentionMask,
        headMask: headMask,
        encoderHiddenStates: encoderHiddenStates,
        // TODO encoder_attention_mask
        outputAttentions: outputCrossAttention,
      );
      embeddings = residual + crossAttentionOutput;
    }

    residual = embeddings;
    embeddings = ln2.forward(embeddings, context: context);
    embeddings = mlp.forward(embeddings, context: context);
    embeddings = embeddings + residual;

    return embeddings;
  }

  void resetKeyValueCache({Tensor? key, Tensor? value}) {
    attention.keyValueCache.reset(key: key, value: value);
  }

  int get layerIdx => attention.layerIdx;

  int get embedDim => attention.embedDim;

  @override
  void resetParameters() {
    ln1.resetParameters();
    attention.resetParameters();
    ln2.resetParameters();
    mlp.resetParameters();
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  Map<String, dynamic> get meta => {};

  @override
  late final Iterable<Module> submodules = [
    ln1,
    attention,
    ln2,
    mlp,
    if (crossAttention != null) crossAttention!,
    if (lnCrossAttention != null) lnCrossAttention!,
  ];

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
    required int? mlpInnerDim,
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required bool isCrossAttention,
    required int maxPositionEmbeddings,
    required Activation activation,
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
      isCrossAttention: false,
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
      activation: activation,
      residualDropoutProbability: residualDropoutProbability,
    );

    GPT2Attention? crossAttention;
    LayerNorm? lnCrossAttention;
    if (isCrossAttention) {
      crossAttention = GPT2Attention.make(
        name: 'crossattention',
        isCrossAttention: true,
        layerIdx: layerIdx,
        embedDim: embedDim,
        attentionDropoutProbability: attentionDropoutProbability,
        residualDropoutProbability: residualDropoutProbability,
        numHeads: numHeads,
        scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
        maxPositionEmbeddings: maxPositionEmbeddings,
      );
      lnCrossAttention = LayerNorm(
        normalizedShape: [embedDim],
        eps: layerNormEpsilon,
      );
    }

    return GPT2Block(
      name: name,
      ln1: ln1,
      attention: attn,
      ln2: ln2,
      mlp: mlp,
      crossAttention: crossAttention,
      lnCrossAttention: lnCrossAttention,
    );
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
    String crossAttentionName = 'crossattention',
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required int numHeads,
    required bool isCrossAttention,
    required int layerIdx,
    required bool scaleAttnByInverseLayerIdx,
    required int maxPositionEmbeddings,
    required Activation activation,
    required int embedDim,
  }) async {
    final attn = await GPT2Attention.loadFromSafeTensor(
      loader,
      prefix: '$prefix$attentionName.',
      name: attentionName,
      attentionDropoutProbability: attentionDropoutProbability,
      residualDropoutProbability: residualDropoutProbability,
      isCrossAttention: false,
      numHeads: numHeads,
      layerIdx: layerIdx,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );
    // final embedDim = attn.embedDim;
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
      activation: activation,
      residualDropoutProbability: residualDropoutProbability,
    );

    GPT2Attention? crossAttention;
    LayerNorm? lnCrossAttention;
    if (isCrossAttention) {
      crossAttention = await GPT2Attention.loadFromSafeTensor(
        loader,
        prefix: '$prefix$crossAttentionName.',
        name: crossAttentionName,
        layerIdx: layerIdx,
        attentionDropoutProbability: attentionDropoutProbability,
        residualDropoutProbability: residualDropoutProbability,
        numHeads: numHeads,
        scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
        maxPositionEmbeddings: maxPositionEmbeddings,
        isCrossAttention: true,
      );
    }

    return GPT2Block(
      name: name,
      ln1: ln1,
      attention: attn,
      ln2: ln2,
      mlp: mlp,
      crossAttention: crossAttention,
      lnCrossAttention: lnCrossAttention,
    );
  }
}
