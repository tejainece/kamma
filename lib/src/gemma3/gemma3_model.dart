import 'dart:math' as math;
import 'package:kamma/kamma.dart';
import 'package:tensor/tensor.dart';
import 'gemma3_config.dart';

// -----------------------------------------------------------------------------
// Gemma3RMSNorm
// -----------------------------------------------------------------------------
class Gemma3RMSNorm extends Module implements SimpleModule {
  final double eps;
  final Tensor weight; // This is a parameter

  Gemma3RMSNorm({required super.name, required this.eps, required this.weight});

  static Gemma3RMSNorm make({
    required String name,
    required int dim,
    double eps = 1e-6,
  }) {
    return Gemma3RMSNorm(name: name, eps: eps, weight: Tensor.zeros([dim]));
  }

  static Future<Gemma3RMSNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int dim,
    required double eps,
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    return Gemma3RMSNorm(name: name, eps: eps, weight: weight);
  }

  @override
  Tensor forward(Tensor embeddings, {required Context context}) {
    final xFloat = embeddings.to(dataType: DataType.float32);
    final variance = xFloat.pow(2).mean(dim: [-1], keepDim: true);
    final rsqrt = (variance + eps).rsqrt();
    final normX = xFloat * rsqrt;

    final weightFloat = weight.to(dataType: DataType.float32);
    final scaled = normX * (weightFloat + 1.0);

    return scaled.to(dataType: embeddings.dataType);
  }

  @override
  Iterable<Tensor> get parameters => [weight];

  @override
  void resetParameters() {
    weight.fill_(0.0);
  }

  @override
  late final Iterable<Module> submodules = [];

  @override
  Map<String, dynamic> get meta => {"eps": eps, "dim": weight.shape[0]};
}

// -----------------------------------------------------------------------------
// Gemma3RotaryEmbedding
// -----------------------------------------------------------------------------
class Gemma3RotaryEmbedding {
  late final Tensor invFreq;
  late final double attentionScaling;
  final Gemma3Config config;

  Gemma3RotaryEmbedding(this.config) {
    _init();
  }

  void _init() {
    _initDefault();
  }

  void _initDefault() {
    final base = config.ropeTheta;
    final headDim = config.headDim;
    final dim = headDim;

    final indices = Tensor.arange(0, dim, step: 2, datatype: DataType.float32);
    final exponent = indices / dim;

    final lnBase = math.log(base);
    final denom = (exponent * lnBase).exp();

    invFreq = Tensor.full([1], 1.0, datatype: DataType.float32) / denom;
    attentionScaling = 1.0;
  }

  (Tensor, Tensor) call(Tensor x, Tensor positionIds) {
    final positionIdsExpanded = positionIds.unsqueeze(1); // [B, 1, S]
    final posIdsFloat = positionIdsExpanded.to(dataType: DataType.float32);

    var currentInvFreq = invFreq;
    if (currentInvFreq.device.deviceType != x.device.deviceType ||
        currentInvFreq.device.deviceIndex != x.device.deviceIndex) {
      currentInvFreq = currentInvFreq.to(device: x.device);
    }

    final invFreqExpanded = currentInvFreq
        .unsqueeze(0)
        .unsqueeze(-1); // [1, D/2, 1]

    final freqs = invFreqExpanded.matmul(posIdsFloat).transpose(1, 2);

    final emb = Tensor.cat([freqs, freqs], dim: -1); // [B, S, D]

    final cos = emb.cos() * attentionScaling;
    final sin = emb.sin() * attentionScaling;

    return (cos.to(dataType: x.dataType), sin.to(dataType: x.dataType));
  }
}

// -----------------------------------------------------------------------------
// Helper Functions for RoPE
// -----------------------------------------------------------------------------
Tensor rotateHalf(Tensor x) {
  final lastDim = x.shape.last;
  final halfDim = lastDim ~/ 2;
  final x1 = x.slice(-1, 0, end: halfDim);
  final x2 = x.slice(-1, halfDim, end: lastDim);
  final negX2 = x2 * -1.0;
  return Tensor.cat([negX2, x1], dim: -1);
}

(Tensor, Tensor) applyRotaryPosEmb(
  Tensor q, // [B, H, S, D]
  Tensor k, // [B, H_kv, S, D]
  Tensor cos, // [B, S, D]
  Tensor sin, { // [B, S, D]
  int unsqueezeDim = 1,
}) {
  final cosUnsq = cos.unsqueeze(unsqueezeDim);
  final sinUnsq = sin.unsqueeze(unsqueezeDim);

  final qEmbed = (q * cosUnsq) + (rotateHalf(q) * sinUnsq);
  final kEmbed = (k * cosUnsq) + (rotateHalf(k) * sinUnsq);

  return (qEmbed, kEmbed);
}

// -----------------------------------------------------------------------------
// Gemma3MLP
// -----------------------------------------------------------------------------
class Gemma3MLP extends Module implements SimpleModule {
  final LinearLayer gateProj;
  final LinearLayer upProj;
  final LinearLayer downProj;
  final Activation activation;

  Gemma3MLP({
    required super.name,
    required this.gateProj,
    required this.upProj,
    required this.downProj,
    required this.activation,
  });

  static Gemma3MLP make({
    required String name,
    required int hiddenSize,
    required int intermediateSize,
    required Activation activation,
  }) {
    final gateProj = LinearLayer.make(
      name: 'gate_proj',
      inFeatures: hiddenSize,
      outFeatures: intermediateSize,
      hasBias: false,
    );
    final upProj = LinearLayer.make(
      name: 'up_proj',
      inFeatures: hiddenSize,
      outFeatures: intermediateSize,
      hasBias: false,
    );
    final downProj = LinearLayer.make(
      name: 'down_proj',
      inFeatures: intermediateSize,
      outFeatures: hiddenSize,
      hasBias: false,
    );

    return Gemma3MLP(
      name: name,
      gateProj: gateProj,
      upProj: upProj,
      downProj: downProj,
      activation: activation,
    );
  }

  static Future<Gemma3MLP> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required Activation activation,
  }) async {
    final gateProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}gate_proj.',
      name: 'gate_proj',
    );
    final upProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}up_proj.',
      name: 'up_proj',
    );
    final downProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}down_proj.',
      name: 'down_proj',
    );

    return Gemma3MLP(
      name: name,
      gateProj: gateProj,
      upProj: upProj,
      downProj: downProj,
      activation: activation,
    );
  }

  @override
  Tensor forward(Tensor embeddings, {required Context context}) {
    context.onloadModule(this);

    final gate = gateProj.forward(embeddings, context: context);
    final act = activation.forward(gate, context: context);
    final up = upProj.forward(embeddings, context: context);

    final fused = act * up;
    final output = downProj.forward(fused, context: context);

    return output;
  }

  @override
  void resetParameters() {
    gateProj.resetParameters();
    upProj.resetParameters();
    downProj.resetParameters();
  }

  @override
  Iterable<Tensor> get parameters => [];

  @override
  late final Iterable<Module> submodules = [gateProj, upProj, downProj];

  @override
  Map<String, dynamic> get meta => {};
}

// -----------------------------------------------------------------------------
// Gemma3Attention
// -----------------------------------------------------------------------------
class Gemma3Attention extends Module {
  final Gemma3Config config;
  final int layerIdx;
  final int headDim;
  final int numHeads;
  final int numKeyValueHeads;
  final double scaling;
  final bool isSliding;
  final int? slidingWindow;

  final LinearLayer qProj;
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer oProj;

  final Gemma3RMSNorm qNorm;
  final Gemma3RMSNorm kNorm;

  Gemma3Attention({
    required super.name,
    required this.config,
    required this.layerIdx,
    required this.qProj,
    required this.kProj,
    required this.vProj,
    required this.oProj,
    required this.qNorm,
    required this.kNorm,
  }) : headDim = config.headDim,
       numHeads = config.numAttentionHeads,
       numKeyValueHeads = config.numKeyValueHeads,
       scaling = math.pow(config.queryPreAttnScalar, -0.5).toDouble(),
       isSliding = config.layerTypes?[layerIdx] == "sliding_attention",
       slidingWindow = (config.layerTypes?[layerIdx] == "sliding_attention")
           ? config.slidingWindow
           : null;

  static Gemma3Attention make({
    required String name,
    required Gemma3Config config,
    required int layerIdx,
  }) {
    final headDim = config.headDim;
    return Gemma3Attention(
      name: name,
      config: config,
      layerIdx: layerIdx,
      qProj: LinearLayer.make(
        name: 'q_proj',
        inFeatures: config.hiddenSize,
        outFeatures: config.numAttentionHeads * headDim,
        hasBias: config.attentionBias,
      ),
      kProj: LinearLayer.make(
        name: 'k_proj',
        inFeatures: config.hiddenSize,
        outFeatures: config.numKeyValueHeads * headDim,
        hasBias: config.attentionBias,
      ),
      vProj: LinearLayer.make(
        name: 'v_proj',
        inFeatures: config.hiddenSize,
        outFeatures: config.numKeyValueHeads * headDim,
        hasBias: config.attentionBias,
      ),
      oProj: LinearLayer.make(
        name: 'o_proj',
        inFeatures: config.numAttentionHeads * headDim,
        outFeatures: config.hiddenSize,
        hasBias: config.attentionBias,
      ),
      qNorm: Gemma3RMSNorm.make(
        name: 'q_norm',
        dim: headDim,
        eps: config.rmsNormEps,
      ),
      kNorm: Gemma3RMSNorm.make(
        name: 'k_norm',
        dim: headDim,
        eps: config.rmsNormEps,
      ),
    );
  }

  static Future<Gemma3Attention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required Gemma3Config config,
    required int layerIdx,
  }) async {
    final headDim = config.headDim;
    final qProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}q_proj.',
      name: 'q_proj',
    );
    final kProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}k_proj.',
      name: 'k_proj',
    );
    final vProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}v_proj.',
      name: 'v_proj',
    );
    final oProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}o_proj.',
      name: 'o_proj',
    );

    final qNorm = await Gemma3RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}q_norm.',
      name: 'q_norm',
      dim: headDim,
      eps: config.rmsNormEps,
    );
    final kNorm = await Gemma3RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}k_norm.',
      name: 'k_norm',
      dim: headDim,
      eps: config.rmsNormEps,
    );

    return Gemma3Attention(
      name: name,
      config: config,
      layerIdx: layerIdx,
      qProj: qProj,
      kProj: kProj,
      vProj: vProj,
      oProj: oProj,
      qNorm: qNorm,
      kNorm: kNorm,
    );
  }

  ({Tensor output, Tensor? attentionWeights, List<Tensor>? pastKeyValues})
  forward(
    Tensor hiddenStates, {
    required (Tensor, Tensor) rotaryEmbeddings,
    Tensor? attentionMask,
    List<Tensor>? pastKeyValues,
    bool useCache = false,
    bool outputAttentions = false,
    required Context context,
  }) {
    context.onloadModule(this);

    final inputShape = hiddenStates.shape;
    final batchSize = inputShape[0];
    final seqLength = inputShape[1];

    // Project Q, K, V
    var q = qProj.forward(hiddenStates, context: context);
    var k = kProj.forward(hiddenStates, context: context);
    var v = vProj.forward(hiddenStates, context: context);

    // Reshape Q, K, V
    q = q.view([batchSize, seqLength, numHeads, headDim]).transpose(1, 2);
    k = k
        .view([batchSize, seqLength, numKeyValueHeads, headDim])
        .transpose(1, 2);
    v = v
        .view([batchSize, seqLength, numKeyValueHeads, headDim])
        .transpose(1, 2);

    // Apply norm to Q and K
    q = qNorm.forward(q, context: context);
    k = kNorm.forward(k, context: context);

    // Apply RoPE
    final (cos, sin) = rotaryEmbeddings;
    final (qRot, kRot) = applyRotaryPosEmb(q, k, cos, sin);
    q = qRot;
    k = kRot;

    // KV Cache update
    if (pastKeyValues != null && useCache) {
      // Concatenate with past key values
      final pastK = pastKeyValues[0];
      final pastV = pastKeyValues[1];
      k = Tensor.cat([pastK, k], dim: 2);
      v = Tensor.cat([pastV, v], dim: 2);
    }

    final currentKeyValues = useCache ? [k, v] : null;

    // Grouped Query Attention
    final numKeyValueGroups = numHeads ~/ numKeyValueHeads;
    if (numKeyValueGroups > 1) {
      final kvSeqLen = k.shape[2];
      k = k
          .unsqueeze(2)
          .expand([
            batchSize,
            numKeyValueHeads,
            numKeyValueGroups,
            kvSeqLen,
            headDim,
          ])
          .reshape([batchSize, numHeads, kvSeqLen, headDim]);
      v = v
          .unsqueeze(2)
          .expand([
            batchSize,
            numKeyValueHeads,
            numKeyValueGroups,
            kvSeqLen,
            headDim,
          ])
          .reshape([batchSize, numHeads, kvSeqLen, headDim]);
    }

    // Scaled Dot Product Attention
    var attnWeights = q.matmul(k.transpose(-1, -2)) * scaling;

    // Softcapping
    if (config.attnLogitSoftcapping != null) {
      final softcap = config.attnLogitSoftcapping!;
      attnWeights = attnWeights / softcap;
      attnWeights = attnWeights.tanh();
      attnWeights = attnWeights * softcap;
    }

    if (attentionMask != null) {
      attnWeights = attnWeights + attentionMask;
    }

    // Softmax
    attnWeights = attnWeights.softmax(-1);

    // Output
    var attnOutput = attnWeights.matmul(v);
    attnOutput = attnOutput.transpose(1, 2).contiguous().reshape([
      batchSize,
      seqLength,
      numHeads * headDim,
    ]);

    attnOutput = oProj.forward(attnOutput, context: context);

    return (
      output: attnOutput,
      attentionWeights: outputAttentions ? attnWeights : null,
      pastKeyValues: currentKeyValues,
    );
  }

  @override
  void resetParameters() {
    qProj.resetParameters();
    kProj.resetParameters();
    vProj.resetParameters();
    oProj.resetParameters();
    qNorm.resetParameters();
    kNorm.resetParameters();
  }

  @override
  late final Iterable<Module> submodules = [
    qProj,
    kProj,
    vProj,
    oProj,
    qNorm,
    kNorm,
  ];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  Map<String, dynamic> get meta => {};
}

// -----------------------------------------------------------------------------
// Gemma3DecoderLayer
// -----------------------------------------------------------------------------
class Gemma3DecoderLayer extends Module {
  final Gemma3Config config;
  final int layerIdx;
  final Gemma3Attention selfAttn;
  final Gemma3MLP mlp;
  final Gemma3RMSNorm inputLayernorm;
  final Gemma3RMSNorm postAttentionLayernorm;
  final Gemma3RMSNorm preFeedforwardLayernorm;
  final Gemma3RMSNorm postFeedforwardLayernorm;

  Gemma3DecoderLayer({
    required super.name,
    required this.config,
    required this.layerIdx,
    required this.selfAttn,
    required this.mlp,
    required this.inputLayernorm,
    required this.postAttentionLayernorm,
    required this.preFeedforwardLayernorm,
    required this.postFeedforwardLayernorm,
  });

  static Gemma3DecoderLayer make({
    required Gemma3Config config,
    required int layerIdx,
  }) {
    return Gemma3DecoderLayer(
      name: 'layers.\$layerIdx',
      config: config,
      layerIdx: layerIdx,
      selfAttn: Gemma3Attention.make(
        name: 'self_attn',
        config: config,
        layerIdx: layerIdx,
      ),
      mlp: Gemma3MLP.make(
        name: 'mlp',
        hiddenSize: config.hiddenSize,
        intermediateSize: config.intermediateSize,
        activation: Activation.gelu,
      ),
      inputLayernorm: Gemma3RMSNorm.make(
        name: 'input_layernorm',
        dim: config.hiddenSize,
        eps: config.rmsNormEps,
      ),
      postAttentionLayernorm: Gemma3RMSNorm.make(
        name: 'post_attention_layernorm',
        dim: config.hiddenSize,
        eps: config.rmsNormEps,
      ),
      preFeedforwardLayernorm: Gemma3RMSNorm.make(
        name: 'pre_feedforward_layernorm',
        dim: config.hiddenSize,
        eps: config.rmsNormEps,
      ),
      postFeedforwardLayernorm: Gemma3RMSNorm.make(
        name: 'post_feedforward_layernorm',
        dim: config.hiddenSize,
        eps: config.rmsNormEps,
      ),
    );
  }

  static Future<Gemma3DecoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required Gemma3Config config,
    required int layerIdx,
    required String prefix,
  }) async {
    final selfAttn = await Gemma3Attention.loadFromSafeTensor(
      loader,
      prefix: '${prefix}self_attn.',
      name: 'self_attn',
      config: config,
      layerIdx: layerIdx,
    );
    final mlp = await Gemma3MLP.loadFromSafeTensor(
      loader,
      prefix: '${prefix}mlp.',
      name: 'mlp',
      activation: Activation.gelu,
    );

    final inputLayernorm = await Gemma3RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}input_layernorm.',
      name: 'input_layernorm',
      dim: config.hiddenSize,
      eps: config.rmsNormEps,
    );
    final postAttentionLayernorm = await Gemma3RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}post_attention_layernorm.',
      name: 'post_attention_layernorm',
      dim: config.hiddenSize,
      eps: config.rmsNormEps,
    );
    final preFeedforwardLayernorm = await Gemma3RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}pre_feedforward_layernorm.',
      name: 'pre_feedforward_layernorm',
      dim: config.hiddenSize,
      eps: config.rmsNormEps,
    );
    final postFeedforwardLayernorm = await Gemma3RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}post_feedforward_layernorm.',
      name: 'post_feedforward_layernorm',
      dim: config.hiddenSize,
      eps: config.rmsNormEps,
    );

    return Gemma3DecoderLayer(
      name: 'layers.\$layerIdx',
      config: config,
      layerIdx: layerIdx,
      selfAttn: selfAttn,
      mlp: mlp,
      inputLayernorm: inputLayernorm,
      postAttentionLayernorm: postAttentionLayernorm,
      preFeedforwardLayernorm: preFeedforwardLayernorm,
      postFeedforwardLayernorm: postFeedforwardLayernorm,
    );
  }

  ({Tensor output, Tensor? attentionWeights, List<Tensor>? pastKeyValues})
  forward(
    Tensor hiddenStates, {
    required Context context,
    required (Tensor, Tensor) rotaryEmbeddings,
    Tensor? attentionMask,
    List<Tensor>? pastKeyValues,
    bool useCache = false,
    bool outputAttentions = false,
  }) {
    context.onloadModule(this);

    var residual = hiddenStates;
    hiddenStates = inputLayernorm.forward(hiddenStates, context: context);

    final attnResult = selfAttn.forward(
      hiddenStates,
      rotaryEmbeddings: rotaryEmbeddings,
      attentionMask: attentionMask,
      pastKeyValues: pastKeyValues,
      useCache: useCache,
      outputAttentions: outputAttentions,
      context: context,
    );

    hiddenStates = attnResult.output;
    hiddenStates = postAttentionLayernorm.forward(
      hiddenStates,
      context: context,
    );
    hiddenStates = residual + hiddenStates;

    residual = hiddenStates;
    hiddenStates = preFeedforwardLayernorm.forward(
      hiddenStates,
      context: context,
    );
    hiddenStates = mlp.forward(hiddenStates, context: context);
    hiddenStates = postFeedforwardLayernorm.forward(
      hiddenStates,
      context: context,
    );
    hiddenStates = residual + hiddenStates;

    return (
      output: hiddenStates,
      attentionWeights: attnResult.attentionWeights,
      pastKeyValues: attnResult.pastKeyValues,
    );
  }

  @override
  void resetParameters() {
    selfAttn.resetParameters();
    mlp.resetParameters();
    inputLayernorm.resetParameters();
    postAttentionLayernorm.resetParameters();
    preFeedforwardLayernorm.resetParameters();
    postFeedforwardLayernorm.resetParameters();
  }

  @override
  Iterable<Tensor> get parameters => [];

  @override
  late final Iterable<Module> submodules = [
    selfAttn,
    mlp,
    inputLayernorm,
    postAttentionLayernorm,
    preFeedforwardLayernorm,
    postFeedforwardLayernorm,
  ];

  @override
  Map<String, dynamic> get meta => {};
}

// -----------------------------------------------------------------------------
// Gemma3Model
// -----------------------------------------------------------------------------
class Gemma3Model extends Module {
  final Gemma3Config config;
  final EmbeddingLayer embedTokens;
  final List<Gemma3DecoderLayer> layers;
  final Gemma3RMSNorm norm;
  final Gemma3RotaryEmbedding rotaryEmb;

  Gemma3Model({
    required super.name,
    required this.config,
    required this.embedTokens,
    required this.layers,
    required this.norm,
  }) : rotaryEmb = Gemma3RotaryEmbedding(config);

  static Gemma3Model make({
    required String name,
    required Gemma3Config config,
  }) {
    return Gemma3Model(
      name: name,
      config: config,
      embedTokens: EmbeddingLayer.make(
        name: 'embed_tokens',
        numEmbeddings: config.vocabSize,
        embedDim: config.hiddenSize,
        paddingIdx: config.padTokenId,
      ),
      layers: List.generate(
        config.numHiddenLayers,
        (i) => Gemma3DecoderLayer.make(config: config, layerIdx: i),
      ),
      norm: Gemma3RMSNorm.make(
        name: 'norm',
        dim: config.hiddenSize,
        eps: config.rmsNormEps,
      ),
    );
  }

  static Future<Gemma3Model> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required Gemma3Config config,
  }) async {
    final embedTokens = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}embed_tokens.',
      name: 'embed_tokens',
      paddingIdx: config.padTokenId,
    );

    final layers = <Gemma3DecoderLayer>[];
    for (int i = 0; i < config.numHiddenLayers; i++) {
      layers.add(
        await Gemma3DecoderLayer.loadFromSafeTensor(
          loader,
          config: config,
          layerIdx: i,
          prefix: '${prefix}layers.\$i.',
        ),
      );
    }

    final norm = await Gemma3RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}norm.',
      name: 'norm',
      dim: config.hiddenSize,
      eps: config.rmsNormEps,
    );

    return Gemma3Model(
      name: name,
      config: config,
      embedTokens: embedTokens,
      layers: layers,
      norm: norm,
    );
  }

  ({
    Tensor lastHiddenState,
    List<List<Tensor>>? pastKeyValues,
    List<Tensor>? hiddenStates,
    List<Tensor>? attentions,
  })
  forward(
    Tensor inputIds, {
    required Context context,
    Tensor? attentionMask,
    List<List<Tensor>>? pastKeyValues,
    bool useCache = false,
    bool outputAttentions = false,
    bool outputHiddenStates = false,
  }) {
    context.onloadModule(this);

    // TODO: embeddings processing similar to PyTorch
    var hiddenStates = embedTokens.forward(inputIds, context: context);

    // Scale embeddings
    final normalizer = math.sqrt(config.hiddenSize);
    hiddenStates = hiddenStates * normalizer;

    // TODO: RoPE position embeddings
    final seqLen = hiddenStates.shape[1];
    final device = hiddenStates.device;
    final positionIds = Tensor.arange(
      0,
      seqLen,
      datatype: DataType.int64,
      device: device,
    ).unsqueeze(0);
    final rotaryEmbeddings = rotaryEmb(hiddenStates, positionIds);

    final allHiddenStates = outputHiddenStates ? <Tensor>[] : null;
    final allAttentions = outputAttentions ? <Tensor>[] : null;
    final nextDecoderCache = useCache ? <List<Tensor>>[] : null;

    for (int i = 0; i < layers.length; i++) {
      if (outputHiddenStates) {
        allHiddenStates!.add(hiddenStates);
      }

      final layerOutputs = layers[i].forward(
        hiddenStates,
        context: context,
        rotaryEmbeddings: rotaryEmbeddings,
        attentionMask: attentionMask,
        pastKeyValues: pastKeyValues?[i],
        useCache: useCache,
        outputAttentions: outputAttentions,
      );

      hiddenStates = layerOutputs.output;

      if (outputAttentions) {
        allAttentions!.add(layerOutputs.attentionWeights!);
      }
      if (useCache) {
        nextDecoderCache!.add(layerOutputs.pastKeyValues!);
      }
    }

    hiddenStates = norm.forward(hiddenStates, context: context);

    if (outputHiddenStates) {
      allHiddenStates!.add(hiddenStates);
    }

    return (
      lastHiddenState: hiddenStates,
      pastKeyValues: nextDecoderCache,
      hiddenStates: allHiddenStates,
      attentions: allAttentions,
    );
  }

  @override
  void resetParameters() {
    embedTokens.resetParameters();
    for (final layer in layers) {
      layer.resetParameters();
    }
    norm.resetParameters();
  }

  @override
  Iterable<Tensor> get parameters => [];

  @override
  late final Iterable<Module> submodules = [embedTokens, ...layers, norm];

  @override
  Map<String, dynamic> get meta => {};
}

// -----------------------------------------------------------------------------
// Gemma3ForCausalLM
// -----------------------------------------------------------------------------
class Gemma3ForCausalLM extends Module {
  final Gemma3Config config;
  final Gemma3Model model;
  final LinearLayer lmHead;
  final double? finalLogitSoftcapping;

  Gemma3ForCausalLM({
    required super.name,
    required this.config,
    required this.model,
    required this.lmHead,
  }) : finalLogitSoftcapping = config.finalLogitSoftcapping;

  static Gemma3ForCausalLM make({
    required String name,
    required Gemma3Config config,
  }) {
    return Gemma3ForCausalLM(
      name: name,
      config: config,
      model: Gemma3Model.make(name: 'model', config: config),
      lmHead: LinearLayer.make(
        name: 'lm_head',
        inFeatures: config.hiddenSize,
        outFeatures: config.vocabSize,
        hasBias: false,
      ),
    );
  }

  static Future<Gemma3ForCausalLM> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required Gemma3Config config,
  }) async {
    final model = await Gemma3Model.loadFromSafeTensor(
      loader,
      prefix: '${prefix}model.',
      name: 'model',
      config: config,
    );
    final lmHead = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}lm_head.',
      name: 'lm_head',
    );

    return Gemma3ForCausalLM(
      name: name,
      config: config,
      model: model,
      lmHead: lmHead,
    );
  }

  ({
    Tensor logits,
    List<List<Tensor>>? pastKeyValues,
    List<Tensor>? hiddenStates,
    List<Tensor>? attentions,
  })
  forward(
    Tensor inputIds, {
    required Context context,
    Tensor? attentionMask,
    List<List<Tensor>>? pastKeyValues,
    bool useCache = false,
    bool outputAttentions = false,
    bool outputHiddenStates = false,
  }) {
    context.onloadModule(this);

    final outputs = model.forward(
      inputIds,
      context: context,
      attentionMask: attentionMask,
      pastKeyValues: pastKeyValues,
      useCache: useCache,
      outputAttentions: outputAttentions,
      outputHiddenStates: outputHiddenStates,
    );

    var hiddenStates = outputs.lastHiddenState;
    var logits = lmHead.forward(hiddenStates, context: context);

    if (finalLogitSoftcapping != null) {
      logits = logits / finalLogitSoftcapping!;
      logits = logits.tanh();
      logits = logits * finalLogitSoftcapping!;
    }

    return (
      logits: logits,
      pastKeyValues: outputs.pastKeyValues,
      hiddenStates: outputs.hiddenStates,
      attentions: outputs.attentions,
    );
  }

  @override
  void resetParameters() {
    model.resetParameters();
    lmHead.resetParameters();
  }

  @override
  Iterable<Tensor> get parameters => [];

  @override
  late final Iterable<Module> submodules = [model, lmHead];

  @override
  Map<String, dynamic> get meta => {};
}
