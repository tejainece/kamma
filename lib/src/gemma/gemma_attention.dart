import 'dart:math' as math;
import 'package:kamma/kamma.dart';
import 'package:kamma/src/gpt2/gpt2_attention.dart'; // For AttentionCache
import 'gemma_rotary_embedding.dart';
import 'package:tensor/tensor.dart';

class GemmaAttention extends Module {
  final int layerIdx;
  final int numHeads;
  final int numKeyValueHeads;
  final int headDim;
  final int embedDim;
  final bool isCausal;
  final double ropeTheta;

  final LinearLayer qProj;
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer oProj;

  final Dropout attentionDropout;
  late final GPT2AttentionMethod attentionMethod;
  late final AttentionCache keyValueCache;

  GemmaAttention({
    required super.name,
    required this.layerIdx,
    required this.numHeads,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.embedDim,
    required this.isCausal,
    required this.ropeTheta,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.attentionDropout,
    required GPT2AttentionMethodType attentionMethodType,
    required int maxPositionEmbeddings,
  }) {
    if (attentionMethodType != GPT2AttentionMethodType.pagedAttention) {
      keyValueCache = AttentionCache.empty();
    } else {
      throw UnimplementedError("Paged Attention is not implemented yet.");
    }

    // Gemma scales Q and K by 1/sqrt(headDim) BEFORE RoPE.
    // And implies no scaling after dot product.
    // So we pass scaleFactor = 1.0 to attentionMethod.

    this.attentionMethod = GPT2AttentionMethod.make(
      attentionMethodType,
      scaleFactor: 1.0,
      isCausal: isCausal,
      attnDropout: attentionDropout,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );
  }

  static GemmaAttention make({
    required String name,
    required int layerIdx,
    required int numHeads,
    required int numKeyValueHeads,
    required int headDim,
    required int embedDim,
    required bool isCausal,
    required double ropeTheta,
    required int maxPositionEmbeddings,
    required bool attentionBias,
    required double attentionDropoutProb,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) {
    return GemmaAttention(
      name: name,
      layerIdx: layerIdx,
      numHeads: numHeads,
      numKeyValueHeads: numKeyValueHeads,
      headDim: headDim,
      embedDim: embedDim,
      isCausal: isCausal,
      ropeTheta: ropeTheta,
      attentionDropout: Dropout(attentionDropoutProb),
      attentionMethodType: attentionMethod,
      maxPositionEmbeddings: maxPositionEmbeddings,
      qProj: LinearLayer.make(
        name: 'q_proj',
        inFeatures: embedDim,
        outFeatures: numHeads * headDim,
        hasBias: attentionBias,
      ),
      kProj: LinearLayer.make(
        name: 'k_proj',
        inFeatures: embedDim,
        outFeatures: numKeyValueHeads * headDim, // MQA/GQA support
        hasBias: attentionBias,
      ),
      vProj: LinearLayer.make(
        name: 'v_proj',
        inFeatures: embedDim,
        outFeatures: numKeyValueHeads * headDim,
        hasBias: attentionBias,
      ),
      oProj: LinearLayer.make(
        name: 'o_proj',
        inFeatures: numHeads * headDim,
        outFeatures: embedDim,
        hasBias: attentionBias,
      ),
    );
  }

  static Future<GemmaAttention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int layerIdx,
    required int numHeads,
    required int numKeyValueHeads,
    required int headDim,
    required int embedDim,
    required bool isCausal,
    required double ropeTheta,
    required int maxPositionEmbeddings,
    required double attentionDropoutProb,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) async {
    return GemmaAttention(
      name: name,
      layerIdx: layerIdx,
      numHeads: numHeads,
      numKeyValueHeads: numKeyValueHeads,
      headDim: headDim,
      embedDim: embedDim,
      isCausal: isCausal,
      ropeTheta: ropeTheta,
      attentionDropout: Dropout(attentionDropoutProb),
      attentionMethodType: attentionMethod,
      maxPositionEmbeddings: maxPositionEmbeddings,
      qProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}q_proj.',
        name: 'q_proj',
      ),
      kProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}k_proj.',
        name: 'k_proj',
      ),
      vProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}v_proj.',
        name: 'v_proj',
      ),
      oProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}o_proj.',
        name: 'o_proj',
      ),
    );
  }

  Tensor forward(
    Tensor hiddenStates, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    ({Tensor cos, Tensor sin})? positionEmbeddings,
    bool useCache = false,
  }) {
    context.onloadModule(this);

    final batchSize = hiddenStates.shape[0];
    final qLen = hiddenStates.shape[1];

    Tensor queryStates = qProj.forward(hiddenStates, context: context);
    Tensor keyStates = kProj.forward(hiddenStates, context: context);
    Tensor valueStates = vProj.forward(hiddenStates, context: context);

    // Reshape
    queryStates = queryStates
        .view([batchSize, qLen, numHeads, headDim])
        .transpose(1, 2);
    keyStates = keyStates
        .view([batchSize, qLen, numKeyValueHeads, headDim])
        .transpose(1, 2);
    valueStates = valueStates
        .view([batchSize, qLen, numKeyValueHeads, headDim])
        .transpose(1, 2);

    // Scaling Q, K by 1/sqrt(headDim) per Gemma 1 spec
    final scale = 1.0 / math.sqrt(headDim);
    queryStates = queryStates * scale;

    // Apply RoPE
    if (positionEmbeddings != null) {
      final (:cos, :sin) = positionEmbeddings;
      final (:qEmbed, :kEmbed) = GemmaRotaryEmbedding.applyRotaryPosEmb(
        queryStates,
        keyStates,
        cos,
        sin,
      );
      queryStates = qEmbed;
      keyStates = kEmbed;
    }

    if (useCache) {
      keyValueCache.update(newKey: keyStates, newValue: valueStates);
      keyStates = keyValueCache.key;
      valueStates = keyValueCache.value;
    }

    // GQA/MQA repeat
    final numKeyValueGroups = numHeads ~/ numKeyValueHeads;
    if (numKeyValueGroups > 1) {
      keyStates = _repeatKv(keyStates, numKeyValueGroups);
      valueStates = _repeatKv(valueStates, numKeyValueGroups);
    }

    // Attention
    var (:attentionOutput, :attentionWeights) = attentionMethod.perform(
      queryStates,
      keyStates,
      valueStates,
      attentionMask: attentionMask,
      context: context,
    );

    attentionOutput = attentionOutput.transpose(1, 2).contiguous();
    attentionOutput = attentionOutput.view([batchSize, qLen, embedDim]);

    return oProj.forward(attentionOutput, context: context);
  }

  Tensor _repeatKv(Tensor t, int nRep) {
    if (nRep == 1) return t;
    final ds = t.shape; // [batch, numKvHeads, seqLen, headDim]
    return t.unsqueeze(2).expand([ds[0], ds[1], nRep, ds[2], ds[3]]).reshape([
      ds[0],
      ds[1] * nRep,
      ds[2],
      ds[3],
    ]);
  }

  @override
  Iterable<Module> get submodules => [
    qProj,
    kProj,
    vProj,
    oProj,
    attentionMethod,
  ];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    qProj.resetParameters();
    kProj.resetParameters();
    vProj.resetParameters();
    oProj.resetParameters();
    keyValueCache.reset();
  }

  void resetKeyValueCache() {
    keyValueCache.reset();
  }

  @override
  Map<String, dynamic> get meta => {
    'headDim': headDim,
    'numHeads': numHeads,
    'scaleFactor': 1.0 / math.sqrt(headDim), // Actually we apply manually
  };
}
