import 'package:tensor/tensor.dart';
import 'package:kamma/kamma.dart';
import 'deepseek_config.dart';
import 'deepseek_decoder_layer.dart';
import 'deepseek_rms_norm.dart';
import 'deepseek_rotary_embedding.dart';
import 'deepseek_rotary_embedding.dart';

class DeepSeekModel extends Module implements SimpleModule {
  final DeepSeekConfig config;

  late final EmbeddingLayer tokens;
  late final List<DeepSeekDecoderLayer> layers;
  late final DeepSeekRMSNorm norm;
  late final DeepSeekRotaryEmbedding rotaryEmb;

  DeepSeekModel({
    required super.name,
    required this.config,
    required this.tokens,
    required this.layers,
    required this.norm,
    required this.rotaryEmb,
  });

  static DeepSeekModel make(DeepSeekConfig config, {String name = 'model'}) {
    // Shared rotary embedding
    final rotaryEmb = DeepSeekRotaryEmbedding(
      config,
    ); // initDefault handled in constructor

    return DeepSeekModel(
      name: name,
      config: config,
      tokens: EmbeddingLayer.make(
        numEmbeddings: config.vocabSize,
        embedDim: config.hiddenSize,
        name: '${name}.embed_tokens',
        paddingIdx: 0, // Usually 0 or check config
      ),
      layers: List.generate(
        config.numHiddenLayers,
        (i) => DeepSeekDecoderLayer(
          name: '${name}.layers.$i',
          config: config,
          layerIdx: i,
          rotaryEmbedding: rotaryEmb,
        ),
      ),
      norm: DeepSeekRMSNorm(config),
      rotaryEmb: rotaryEmb,
    );
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

    Tensor h = tokens.forward(embeddings, context: context);

    // Position IDs
    if (positionIds == null) {
      final seqLen = embeddings.shape[1];
      if (embeddings.device != null) {
        positionIds = Tensor.arange(
          0,
          seqLen,
          device: embeddings.device,
        ).unsqueeze(0);
      } else {
        positionIds = Tensor.arange(0, seqLen).unsqueeze(0);
      }
    }

    // Precompute Rotary Cos/Sin?
    // DeepSeekV3MLA call rotary embedding internally on v/k.
    // And it needs (cos, sin) OR position_ids.
    // The current DeepSeekV3MLA implementation takes `positionIds` and computes rotation inside.
    // This differs from Llama where it's computed outside.
    // Just consistent with my previous implemenation: pass positionIds.

    for (var layer in layers) {
      h = layer.forward(
        h,
        context: context,
        attentionMask: attentionMask,
        positionIds: positionIds,
        useCache: useCache,
      );
    }

    h = norm.forward(h, context: context);

    return h;
  }

  @override
  Iterable<Module> get submodules => [tokens, ...layers, norm];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  Iterable<Tensor> get nonTrainableParameters => [];

  @override
  void resetParameters() {
    tokens.resetParameters();
    for (var l in layers) l.resetParameters();
    norm.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {'config': config};
}

class DeepSeekForCausalLM extends Module implements SimpleModule {
  final DeepSeekConfig config;
  late final DeepSeekModel model;
  late final LinearLayer lmHead;

  DeepSeekForCausalLM({
    required super.name,
    required this.config,
    required this.model,
    required this.lmHead,
  });

  static DeepSeekForCausalLM make(DeepSeekConfig config) {
    return DeepSeekForCausalLM(
      name: 'deepseek_causal_lm',
      config: config,
      model: DeepSeekModel.make(config, name: 'model'),
      lmHead: LinearLayer.make(
        name: 'lm_head',
        inFeatures: config.hiddenSize,
        outFeatures: config.vocabSize,
        hasBias: false,
      ),
    );
  }

  @override
  Tensor forward(
    Tensor embeddings, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
  }) {
    context.onloadModule(this);

    var h = model.forward(
      embeddings,
      context: context,
      attentionMask: attentionMask,
      positionIds: positionIds,
    );

    var logits = lmHead.forward(h, context: context);
    return logits;
  }

  @override
  Iterable<Module> get submodules => [model, lmHead];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  Iterable<Tensor> get nonTrainableParameters => [];

  @override
  void resetParameters() {
    model.resetParameters();
    lmHead.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {'config': config};
}
