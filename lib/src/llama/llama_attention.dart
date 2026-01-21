import 'dart:math' as math;
import 'package:kamma/kamma.dart';

class LlamaAttention extends Module implements SimpleModule {
  final int layerIdx;
  final int numHeads;

  final double ropeTheta;
  final bool isCausal;

  final Dropout attentionDropout;
  final LinearLayer qProj;
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer oProj;
  late final GPT2AttentionMethod attentionMethod;
  late final AttentionCache keyValueCache;

  LlamaAttention({
    required super.name,
    required this.layerIdx,
    required this.numHeads,
    required this.attentionDropout,
    required int maxPositionEmbeddings,
    required this.ropeTheta,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.isCausal,
    required GPT2AttentionMethodType attentionMethod,
  }) {
    if ((headDim * numHeads) != embedDim) {
      throw Exception('embedDim must be divisible by num_heads');
    }

    if (attentionMethod == GPT2AttentionMethodType.pagedAttention) {
      throw UnimplementedError("Paged Attention is not implemented yet.");
    }
    /* TODO this.attentionMethod = GPT2AttentionMethod.make(
      attentionMethod,
      scaleFactor: scaleFactor,
      isCausal: !isCrossAttention,
      attnDropout: attentionDropout,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );*/
    if (attentionMethod != GPT2AttentionMethodType.pagedAttention) {
      keyValueCache = AttentionCache.empty();
    } else {
      // TODO intialize cache for paged attention
      throw UnimplementedError("Paged Attention is not implemented yet.");
    }
  }

  int get embedDim => qProj.numInFeatures;
  int get headDim => embedDim ~/ numHeads;

  int get numKeyValueHeads => kProj.numOutFeatures ~/ headDim;
  int get numKeyValueGroups => numHeads ~/ numKeyValueHeads;

  @override
  Tensor forward(
    Tensor embeddings, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    ({Tensor cos, Tensor sin})? positionEmbeddings,
    bool useCache = false,
    // Cache object? For now simplistic
  }) {
    context.onloadModule(this);

    final batchSize = embeddings.shape[0];
    final qLen = embeddings.shape[1];

    final queryStates = qProj
        .forward(embeddings, context: context)
        .view([batchSize, qLen, numHeads, headDim])
        .transpose(1, 2);

    final keyStates = kProj
        .forward(embeddings, context: context)
        .view([batchSize, qLen, numKeyValueHeads, headDim])
        .transpose(1, 2);

    Tensor valueStates = vProj
        .forward(embeddings, context: context)
        .view([batchSize, qLen, numKeyValueHeads, headDim])
        .transpose(1, 2);

    // Apply RoPE
    Tensor q = queryStates;
    Tensor k = keyStates;

    if (positionEmbeddings != null) {
      final (:cos, :sin) = positionEmbeddings;
      final (:qEmbed, :kEmbed) = LlamaRotaryEmbedding.applyRotaryPosEmb(
        q,
        k,
        cos,
        sin,
      );
      q = qEmbed;
      k = kEmbed;
    }

    if (useCache) {
      keyValueCache.update(newKey: k, newValue: valueStates);
      k = keyValueCache.key;
      valueStates = keyValueCache.value;
    }

    // Repeat KV if GQA
    if (numKeyValueGroups > 1) {
      k = repeatKv(k, numKeyValueGroups);
      valueStates = repeatKv(
        valueStates,
        numKeyValueGroups,
      ); // v needs repeat too? Yes.
    }
    final v = valueStates;

    // Attention
    // mask shape: (bsz, 1, qLen, kvLen) usually
    // SDPA: softmax(Q @ K.T / sqrt(headDim) + mask) @ V

    final attnWeights = q.matmul(k.transpose(2, 3)) / math.sqrt(headDim);

    Tensor attnScores = attnWeights;
    if (attentionMask != null) {
      // mask is usually additive (0.0 for keep, -inf for mask)
      attnScores = attnScores + attentionMask;
    }

    // softmax
    attnScores = attnScores.softmax(-1).to(dataType: embeddings.dataType);

    // dropout (TODO if training)

    final attnOutput = attnScores.matmul(v);

    final output = attnOutput.transpose(1, 2).contiguous().view([
      batchSize,
      qLen,
      embedDim,
    ]);

    return oProj.forward(output, context: context);
  }

  @override
  Iterable<Module> get submodules => [qProj, kProj, vProj, oProj];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    qProj.resetParameters();
    kProj.resetParameters();
    vProj.resetParameters();
    oProj.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {
    'numHeads': numHeads,
    'headDim': headDim,
    'embedDim': embedDim,
  };

  static LlamaAttention make({
    required String name,
    required int layerIdx,
    required int numHeads,
    required int embedDim,
    required int maxPositionEmbeddings,
    required double ropeTheta,
    required bool hasAttentionBias,
    required int numKeyValueHeads,
    required double attentionDropoutProb,
    required bool isCausal,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
    String qProjName = 'q_proj',
    String kProjName = 'k_proj',
    String vProjName = 'v_proj',
    String oProjName = 'o_proj',
  }) {
    int headDim = embedDim ~/ numHeads;
    return LlamaAttention(
      name: name,
      layerIdx: layerIdx,
      numHeads: numHeads,
      maxPositionEmbeddings: maxPositionEmbeddings,
      ropeTheta: ropeTheta,
      isCausal: isCausal,
      attentionDropout: Dropout(
        attentionDropoutProb,
        name: 'attention_dropout',
      ),
      attentionMethod: attentionMethod,
      qProj: LinearLayer.make(
        name: qProjName,
        inFeatures: embedDim,
        outFeatures: numHeads * headDim,
        hasBias: hasAttentionBias,
      ),
      kProj: LinearLayer.make(
        name: kProjName,
        inFeatures: embedDim,
        outFeatures: numKeyValueHeads * headDim,
        hasBias: hasAttentionBias,
      ),
      vProj: LinearLayer.make(
        name: vProjName,
        inFeatures: embedDim,
        outFeatures: numKeyValueHeads * headDim,
        hasBias: hasAttentionBias,
      ),
      oProj: LinearLayer.make(
        name: oProjName,
        inFeatures: numHeads * headDim,
        outFeatures: embedDim,
        hasBias: hasAttentionBias,
      ),
    );
  }

  static Future<LlamaAttention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int layerIdx,
    required int numHeads,
    required double attentionDropoutProb,
    required int maxPositionEmbeddings,
    required double ropeTheta,
    required bool isCausal,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
    String qProjName = 'q_proj',
    String kProjName = 'k_proj',
    String vProjName = 'v_proj',
    String oProjName = 'o_proj',
  }) async {
    return LlamaAttention(
      layerIdx: layerIdx,
      numHeads: numHeads,
      name: name,
      maxPositionEmbeddings: maxPositionEmbeddings,
      ropeTheta: ropeTheta,
      isCausal: isCausal,
      attentionDropout: Dropout(
        attentionDropoutProb,
        name: 'attention_dropout',
      ),
      attentionMethod: attentionMethod,
      qProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '$prefix$qProjName.',
        name: qProjName,
      ),
      kProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '$prefix$kProjName.',
        name: kProjName,
      ),
      vProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '$prefix$vProjName.',
        name: vProjName,
      ),
      oProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '$prefix$oProjName.',
        name: oProjName,
      ),
    );
  }

  void resetKeyValueCache() {
    keyValueCache.reset();
  }
}

// Helper to repeat KV heads
Tensor repeatKv(Tensor hiddenStates, int nRep) {
  // hiddenStates: (bsz, num_key_value_heads, seqlen, head_dim)
  if (nRep == 1) return hiddenStates;

  final bsz = hiddenStates.shape[0];
  final numKvHeads = hiddenStates.shape[1];
  final seqLen = hiddenStates.shape[2];
  final headDim = hiddenStates.shape[3];

  // Expand: (bsz, num_kv_heads, n_rep, seqlen, head_dim)
  // Then reshape: (bsz, num_kv_heads * n_rep, seqlen, head_dim)

  // Since Tensor.repeat repeats, but here we want interleaving or specific repeat structure.
  // PyTorch: hidden_states[:, :, None, :, :].expand(bsz, num_kv_heads, n_rep, seqlen, head_dim).reshape(...)

  // In Dart Tensor:
  // hiddenStates.unsqueeze(2).expand([bsz, numKvHeads, nRep, seqLen, headDim]).reshape(...)

  return hiddenStates
      .unsqueeze(2)
      .expand([bsz, numKvHeads, nRep, seqLen, headDim])
      .reshape([bsz, numKvHeads * nRep, seqLen, headDim]);
}
