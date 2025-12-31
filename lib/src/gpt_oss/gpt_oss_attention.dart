import 'dart:math';
import 'package:kamma/kamma.dart';
import 'package:kamma/src/gpt_oss/rotrary_embedding.dart';

class GptOssAttention extends Module {
  final int embedDim;
  final int numHeads;
  final int headDim;
  final int numKeyValueHeads;
  final bool isCrossAttention; // Usually false for GPT-OSS
  final int layerIdx;

  // Weights on appropriate device
  final LinearLayer qProj;
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer oProj;

  final GptOssRotaryEmbedding rotaryEmb;
  final Dropout attnDropout;
  final Dropout residDropout;

  GptOssAttention({
    required super.name,
    this.isCrossAttention = false,
    this.layerIdx = 0,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.rotaryEmb,
    required this.attnDropout,
    required this.residDropout,
    required this.embedDim,
    required this.numHeads,
    required int? numKeyValueHeads,
  }) : headDim = embedDim ~/ numHeads,
       // GQA logic
       numKeyValueHeads = numKeyValueHeads ?? numHeads;

  late int numKeyValueGroups = numHeads ~/ numKeyValueHeads;

  // Helper to repeat KV heads
  Tensor _repeatKv(Tensor x, int nRep) {
    if (nRep == 1) return x;

    // x: [batch, num_kv_heads, seq, head_dim]
    final batch = x.shape[0];
    final numKvHeads = x.shape[1];
    final seqLen = x.shape[2];
    final headDim = x.shape[3];

    // expand: [batch, num_kv_heads, nRep, seq, head_dim]
    x = x.unsqueeze(2).expand([batch, numKvHeads, nRep, seqLen, headDim]);

    // reshape: [batch, num_kv_heads * nRep, seq, head_dim]
    return x.reshape([batch, numKvHeads * nRep, seqLen, headDim]);
  }

  Tensor forward(
    Tensor hiddenStates, {
    Tensor? attentionMask,
    Tensor? positionIds,
    List<Tensor>? layerPast,
    bool useCache = false,
    required Context context,
    // Unused args from GPT2 kept for compat if needed
    Tensor? headMask,
    Tensor? encoderHiddenStates,
    Tensor? encoderAttentionMask,
    bool outputAttentions = false,
  }) {
    context.onloadModule(this);

    // 1. Projections
    final queryStates = qProj.forward(hiddenStates, context: context);
    final keyStates = kProj.forward(hiddenStates, context: context);
    final valueStates = vProj.forward(hiddenStates, context: context);

    // 2. Reshape to [batch, seq, heads, dim] -> permute to [batch, heads, seq, dim]
    final batchSize = hiddenStates.shape[0];
    final seqLen = hiddenStates.shape[1];

    var q = queryStates.view([batchSize, seqLen, numHeads, headDim]).permute([
      0,
      2,
      1,
      3,
    ]);
    var k = keyStates
        .view([batchSize, seqLen, numKeyValueHeads, headDim])
        .permute([0, 2, 1, 3]);
    var v = valueStates
        .view([batchSize, seqLen, numKeyValueHeads, headDim])
        .permute([0, 2, 1, 3]);

    // 3. Apply RoPE (inplace on q, k)
    // rotaryEmb.applyRotaryPosEmb(q, k, context: context);

    // Create default position IDs if not provided
    final posIds =
        positionIds ??
        Tensor.arange(
          0,
          seqLen,
          dataType: DataType.int64,
          device: hiddenStates.device, // Crucial: create on same device
        ).unsqueeze(0).expand([batchSize, seqLen]);

    final ropeOutput = rotaryEmb.forward(posIds, seqLen: seqLen);
    GptOssRotaryEmbedding.applyRotaryPosEmb(
      q,
      k,
      ropeOutput.cos,
      ropeOutput.sin,
    );

    // 4. KV Cache handling
    if (layerPast != null) {
      // layerPast assumed to be [k, v] each [batch, heads, past_seq, dim]
      final pastK = layerPast[0];
      final pastV = layerPast[1];
      k = Tensor.cat([pastK, k], dim: 2);
      v = Tensor.cat([pastV, v], dim: 2);
    }

    // 5. Repeat KV for GQA
    k = _repeatKv(k, numKeyValueGroups);
    v = _repeatKv(v, numKeyValueGroups);

    // 6. Attention
    // q: [b, h, s, d], k: [b, h, s_all, d] -> scores: [b, h, s, s_all]
    var scores = q.matmul(k.transpose(-1, -2));
    scores = scores / sqrt(headDim.toDouble());

    if (attentionMask != null) {
      scores = scores + attentionMask;
    }

    var attnWeights = scores.softmax(-1);
    attnWeights = attnDropout.forward(attnWeights, context: context);

    // 7. Output
    var attnOutput = attnWeights.matmul(v); // [b, h, s, d]
    attnOutput = attnOutput.permute([0, 2, 1, 3]).contiguous(); // [b, s, h, d]

    final outputShape = [batchSize, seqLen, embedDim];
    attnOutput = attnOutput.view(outputShape);

    attnOutput = oProj.forward(attnOutput, context: context);
    attnOutput = residDropout.forward(attnOutput, context: context);

    return attnOutput;
  }

  @override
  void resetParameters() {
    qProj.resetParameters();
    kProj.resetParameters();
    vProj.resetParameters();
    oProj.resetParameters();
    // TODO rotaryEmb.populate();
  }

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [
    qProj,
    kProj,
    vProj,
    oProj,
    attnDropout,
    residDropout,
    rotaryEmb,
  ];

  @override
  Map<String, dynamic> get meta => {
    "embedDim": embedDim,
    "numHeads": numHeads,
    "headDim": headDim,
    "numKeyValueHeads": numKeyValueHeads,
    "layerIdx": layerIdx,
  };

  Future<void> copyFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    String qProjName = 'q_proj',
    String kProjName = 'k_proj',
    String vProjName = 'v_proj',
    String oProjName = 'o_proj',
  }) async {
    {
      final w = await loader.loadByName('$prefix$qProjName.weight');
      qProj.weight.copy_(w);
    }
    {
      final w = await loader.loadByName('$prefix$kProjName.weight');
      kProj.weight.copy_(w);
    }
    {
      final w = await loader.loadByName('$prefix$vProjName.weight');
      vProj.weight.copy_(w);
    }
    {
      final w = await loader.loadByName('$prefix$oProjName.weight');
      oProj.weight.copy_(w);
    }
  }

  static Future<GptOssAttention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    String qProjName = 'q_proj',
    String kProjName = 'k_proj',
    String vProjName = 'v_proj',
    String oProjName = 'o_proj',
    required int embedDim,
    required int numHeads,
    required int nPositions,
    required int? numKeyValueHeads,
    required double ropeTheta,
    required double attentionDropoutP,
    required double residDropoutP,
    required bool isCrossAttention,
    required int layerIdx,
  }) async {
    final qProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}q_proj.',
      name: qProjName,
    );
    final kProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}k_proj.',
      name: kProjName,
    );
    final vProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}v_proj.',
      name: vProjName,
    );
    final oProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}o_proj.',
      name: oProjName,
    );
    final headDim = embedDim ~/ numHeads;
    final rotaryEmb = GptOssRotaryEmbedding(
      name: 'rope',
      base: ropeTheta,
      dim: headDim,
      maxPositionEmbeddings: nPositions,
    );

    final attnDropout = Dropout(attentionDropoutP);
    final residDropout = Dropout(residDropoutP);
    return GptOssAttention(
      name: name,
      qProj: qProj,
      kProj: kProj,
      vProj: vProj,
      oProj: oProj,
      rotaryEmb: rotaryEmb,
      attnDropout: attnDropout,
      residDropout: residDropout,
      embedDim: embedDim,
      numHeads: numHeads,
      numKeyValueHeads: numKeyValueHeads,
      isCrossAttention: isCrossAttention,
      layerIdx: layerIdx,
    );
  }

  static GptOssAttention make({
    required String name,
    required bool isCrossAttention,
    required int embedDim,
    required int numHeads,
    required int nPositions,
    required int? numKeyValueHeads,
    required double ropeTheta,
    required double attentionDropoutP,
    required double residDropoutP,
    required int layerIdx,
  }) {
    final headDim = embedDim ~/ numHeads;
    numKeyValueHeads = numKeyValueHeads ?? numHeads;

    final qProj = LinearLayer.make(
      name: '$name.q_proj',
      inFeatures: embedDim,
      outFeatures: numHeads * headDim,
      hasBias: false,
    );

    final kProj = LinearLayer.make(
      name: '$name.k_proj',
      inFeatures: embedDim,
      outFeatures: numKeyValueHeads * headDim,
      hasBias: false,
    );

    final vProj = LinearLayer.make(
      name: '$name.v_proj',
      inFeatures: embedDim,
      outFeatures: numKeyValueHeads * headDim,
      hasBias: false,
    );

    final oProj = LinearLayer.make(
      name: '$name.o_proj',
      inFeatures: numHeads * headDim,
      outFeatures: embedDim,
      hasBias: false,
    );

    final rotaryEmb = GptOssRotaryEmbedding(
      name: '$name.rotary_emb',
      base: ropeTheta,
      dim: headDim,
      maxPositionEmbeddings: nPositions,
    );

    final attnDropout = Dropout(attentionDropoutP);
    final residDropout = Dropout(residDropoutP);

    return GptOssAttention(
      name: name,
      isCrossAttention: isCrossAttention,
      layerIdx: layerIdx,
      qProj: qProj,
      kProj: kProj,
      vProj: vProj,
      oProj: oProj,
      rotaryEmb: rotaryEmb,
      attnDropout: attnDropout,
      residDropout: residDropout,
      embedDim: embedDim,
      numHeads: numHeads,
      numKeyValueHeads: numKeyValueHeads,
    );
  }
}
