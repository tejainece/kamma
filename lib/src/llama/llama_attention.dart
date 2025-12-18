import 'dart:math' as math;
import 'package:tensor/tensor.dart';
import 'llama_config.dart';
import 'llama_rotary_embedding.dart';

class LlamaAttention extends Module implements SimpleModule {
  final LlamaConfig config;
  final int layerIdx;

  late final int hiddenSize;
  late final int numHeads;
  late final int headDim;
  late final int numKeyValueHeads;
  late final int numKeyValueGroups;
  late final int maxPositionEmbeddings;
  late final double ropeTheta;
  late final bool isCausal;
  late final double attentionDropout;

  final LinearLayer qProj;
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer oProj;

  LlamaAttention(
    this.config, {
    this.layerIdx = 0,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
  }) : super(name: 'llama_attention') {
    hiddenSize = config.hiddenSize;
    numHeads = config.numAttentionHeads;
    headDim = config.headDim;
    numKeyValueHeads = config.numKeyValueHeads;
    numKeyValueGroups = numHeads ~/ numKeyValueHeads;
    maxPositionEmbeddings = config.maxPositionEmbeddings;
    ropeTheta = config.ropeTheta;
    isCausal = true; // Llama is causal
    attentionDropout = config.attentionDropout;

    if ((headDim * numHeads) != hiddenSize) {
      throw Exception('hidden_size must be divisible by num_heads');
    }
  }

  static LlamaAttention make(LlamaConfig config, {int layerIdx = 0}) {
    final hiddenSize = config.hiddenSize;
    final numHeads = config.numAttentionHeads;
    final headDim = config.headDim ?? (hiddenSize ~/ numHeads);
    final numKeyValueHeads = config.numKeyValueHeads ?? numHeads;

    return LlamaAttention(
      config,
      layerIdx: layerIdx,
      qProj: LinearLayer.make(
        name: 'q_proj',
        inFeatures: hiddenSize,
        outFeatures: numHeads * headDim,
        hasBias: config.attentionBias,
      ),
      kProj: LinearLayer.make(
        name: 'k_proj',
        inFeatures: hiddenSize,
        outFeatures: numKeyValueHeads * headDim,
        hasBias: config.attentionBias,
      ),
      vProj: LinearLayer.make(
        name: 'v_proj',
        inFeatures: hiddenSize,
        outFeatures: numKeyValueHeads * headDim,
        hasBias: config.attentionBias,
      ),
      oProj: LinearLayer.make(
        name: 'o_proj',
        inFeatures: numHeads * headDim,
        outFeatures: hiddenSize,
        hasBias: config.attentionBias,
      ),
    );
  }

  static Future<LlamaAttention> loadFromSafeTensor(
    SafeTensorLoader loader,
    LlamaConfig config, {
    required String prefix,
    int layerIdx = 0,
  }) async {
    return LlamaAttention(
      config,
      layerIdx: layerIdx,
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

  @override
  Tensor forward(
    Tensor x, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    (Tensor, Tensor)? positionEmbeddings, // (cos, sin)
    bool useCache = false,
    // Cache object? For now simplistic
  }) {
    context.onloadModule(this);

    final bsz = x.shape[0];
    final qLen = x.shape[1];

    final queryStates = qProj
        .forward(x, context: context)
        .view([bsz, qLen, numHeads, headDim])
        .transpose(1, 2);

    final keyStates = kProj
        .forward(x, context: context)
        .view([bsz, qLen, numKeyValueHeads, headDim])
        .transpose(1, 2);

    Tensor valueStates = vProj
        .forward(x, context: context)
        .view([bsz, qLen, numKeyValueHeads, headDim])
        .transpose(1, 2);

    // Apply RoPE
    Tensor q = queryStates;
    Tensor k = keyStates;

    if (positionEmbeddings != null) {
      final (cos, sin) = positionEmbeddings;
      final (qRot, kRot) = applyRotaryPosEmb(q, k, cos, sin);
      q = qRot;
      k = kRot;
    }

    // TODO: KV Caching updates (omitted for initial implementation)

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
    attnScores = attnScores.softmax(-1).to(dataType: x.dataType);

    // dropout (TODO if training)

    final attnOutput = attnScores.matmul(v);

    final output = attnOutput.transpose(1, 2).contiguous().view([
      bsz,
      qLen,
      hiddenSize,
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
    'hiddenSize': hiddenSize,
  };
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
