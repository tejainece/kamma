import 'package:tensor/tensor.dart';
import 'llama_config.dart';
import 'llama_decoder_layer.dart';
import 'llama_rotary_embedding.dart';

class LlamaModel extends Module implements SimpleModule {
  final LlamaConfig config;

  late final EmbeddingLayer tokens; // embed_tokens
  late final List<LlamaDecoderLayer> layers;
  late final RMSNorm norm;
  late final LlamaRotaryEmbedding rotaryEmb;

  LlamaModel(
    this.config, {
    required this.tokens,
    required this.layers,
    required this.norm,
  }) : super(name: 'model') {
    rotaryEmb = LlamaRotaryEmbedding(config);
  }

  static LlamaModel make(LlamaConfig config) {
    return LlamaModel(
      config,
      tokens: EmbeddingLayer.make(
        numEmbeddings: config.vocabSize,
        embedDim: config.hiddenSize,
        name: 'embed_tokens',
        paddingIdx: config.padTokenId,
      ),
      layers: List.generate(
        config.numHiddenLayers,
        (i) => LlamaDecoderLayer.make(config, i),
      ),
      norm: RMSNorm.make(
        normalizedShape: [config.hiddenSize],
        eps: config.rmsNormEps,
        name: 'norm',
      ),
    );
  }

  static Future<LlamaModel> loadFromSafeTensor(
    SafeTensorLoader loader,
    LlamaConfig config,
  ) async {
    final tokens = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      prefix: 'model.embed_tokens.',
      name: 'embed_tokens',
      paddingIdx: config.padTokenId,
    );

    final layers = <LlamaDecoderLayer>[];
    for (int i = 0; i < config.numHiddenLayers; i++) {
      layers.add(
        await LlamaDecoderLayer.loadFromSafeTensor(
          loader,
          config,
          i,
          prefix: 'model.layers.$i.',
        ),
      );
    }

    final norm = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: 'model.norm.',
      name: 'norm',
      normalizedShape: [config.hiddenSize],
      eps: config.rmsNormEps,
    );

    return LlamaModel(config, tokens: tokens, layers: layers, norm: norm);
  }

  @override
  Tensor forward(
    Tensor embeddings, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    bool useCache = false,
  }) {
    context.onloadModule(this);

    // Embeddings
    Tensor hiddenStates = tokens.forward(embeddings, context: context);

    // Prepare RoPE position embeddings
    // (cos, sin) = rotary_emb(hiddenStates, positionIds)
    if (positionIds == null) {
      // Generate pos IDs if null?
      // Usually user provides them or we generate.
      final seqLen = embeddings.shape[1];
      // arange(0, seqLen).unsqueeze(0).expand(bsz, seqLen)
      final device = embeddings.device;
      positionIds = Tensor.arange(0, seqLen, device: device).unsqueeze(0);
    }

    // Compute RoPE cos/sin once for all layers
    final (cos, sin) = rotaryEmb(hiddenStates, positionIds);
    final positionEmbeddings = (cos, sin);

    // Prepare causal mask (if attentionMask is null or if needed)
    // TODO: Causal mask generation logic.
    // For now assuming attentionMask provided or None (if None SDPA might handle if is_causal=True? but usually we need explicit mask).

    for (final layer in layers) {
      hiddenStates = layer.forward(
        hiddenStates,
        context: context,
        attentionMask: attentionMask,
        positionIds: positionIds,
        positionEmbeddings: positionEmbeddings,
        useCache: useCache,
      );
    }

    hiddenStates = norm.forward(hiddenStates, context: context);

    return hiddenStates;
  }

  @override
  Iterable<Module> get submodules => [tokens, ...layers, norm];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    tokens.resetParameters();
    for (final layer in layers) layer.resetParameters();
    norm.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {};
}

class LlamaForCausalLM extends Module implements SimpleModule {
  final LlamaConfig config;
  late final LlamaModel model;
  late final LinearLayer lmHead;

  LlamaForCausalLM(this.config, {required this.model, required this.lmHead})
    : super(name: 'llama');

  static LlamaForCausalLM make(LlamaConfig config) {
    return LlamaForCausalLM(
      config,
      model: LlamaModel.make(config),
      lmHead: LinearLayer.make(
        name: 'lm_head',
        inFeatures: config.hiddenSize,
        outFeatures: config.vocabSize,
        hasBias: false,
      ),
    );
  }

  static Future<LlamaForCausalLM> loadFromSafeTensor(
    SafeTensorLoader loader,
    LlamaConfig config,
  ) async {
    final model = await LlamaModel.loadFromSafeTensor(loader, config);
    final lmHead = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: 'lm_head.',
      name: 'lm_head',
    );
    return LlamaForCausalLM(config, model: model, lmHead: lmHead);
  }

  @override
  Tensor forward(
    Tensor embeddings, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
  }) {
    context.onloadModule(this);

    final hiddenStates = model.forward(
      embeddings,
      context: context,
      attentionMask: attentionMask,
      positionIds: positionIds,
    );

    final logits = lmHead.forward(hiddenStates, context: context);
    return logits;
  }

  @override
  Iterable<Module> get submodules => [model, lmHead];

  @override
  Iterable<Tensor> get parameters => []; // Params in submodules

  @override
  void resetParameters() {
    model.resetParameters();
    lmHead.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {};
}
