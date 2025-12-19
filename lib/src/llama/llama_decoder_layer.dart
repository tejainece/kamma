import 'package:tensor/tensor.dart';
import 'llama_config.dart';
import 'llama_attention.dart';
import 'llama_mlp.dart';

class LlamaDecoderLayer extends Module implements SimpleModule {
  final LlamaConfig config;
  final int layerIdx;

  final LlamaAttention selfAttn;
  final LlamaMLP mlp;
  final RMSNorm inputLayernorm;
  final RMSNorm postAttentionLayernorm;

  LlamaDecoderLayer(
    this.config,
    this.layerIdx, {
    required this.selfAttn,
    required this.mlp,
    required this.inputLayernorm,
    required this.postAttentionLayernorm,
  }) : super(name: 'layers.$layerIdx');

  static LlamaDecoderLayer make(LlamaConfig config, int layerIdx) {
    return LlamaDecoderLayer(
      config,
      layerIdx,
      selfAttn: LlamaAttention.make(config, layerIdx: layerIdx),
      mlp: LlamaMLP.make(config),
      inputLayernorm: RMSNorm.make(
        name: 'input_layernorm',
        normalizedShape: [config.hiddenSize],
        eps: config.rmsNormEps,
      ),
      postAttentionLayernorm: RMSNorm.make(
        name: 'post_attention_layernorm',
        normalizedShape: [config.hiddenSize],
        eps: config.rmsNormEps,
      ),
    );
  }

  static Future<LlamaDecoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader,
    LlamaConfig config,
    int layerIdx, {
    required String prefix,
  }) async {
    return LlamaDecoderLayer(
      config,
      layerIdx,
      selfAttn: await LlamaAttention.loadFromSafeTensor(
        loader,
        config,
        prefix: '${prefix}self_attn.',
        layerIdx: layerIdx,
      ),
      mlp: await LlamaMLP.loadFromSafeTensor(
        loader,
        config,
        prefix: '${prefix}mlp.',
      ),
      inputLayernorm: await RMSNorm.loadFromSafeTensor(
        loader,
        prefix: '${prefix}input_layernorm.',
        name: 'input_layernorm',
        normalizedShape: [config.hiddenSize],
        eps: config.rmsNormEps,
      ),
      postAttentionLayernorm: await RMSNorm.loadFromSafeTensor(
        loader,
        prefix: '${prefix}post_attention_layernorm.',
        name: 'post_attention_layernorm',
        normalizedShape: [config.hiddenSize],
        eps: config.rmsNormEps,
      ),
    );
  }

  @override
  Tensor forward(
    Tensor embeddings, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    (Tensor, Tensor)? positionEmbeddings,
    bool useCache = false,
  }) {
    context.onloadModule(this);

    // Residual connection
    Tensor residual = embeddings;

    // 1. Input Norm
    Tensor hiddenStates = inputLayernorm.forward(embeddings, context: context);

    // 2. Self Attention
    // self_attn(hidden_states, ...)
    hiddenStates = selfAttn.forward(
      hiddenStates,
      context: context,
      attentionMask: attentionMask,
      positionIds: positionIds,
      positionEmbeddings: positionEmbeddings,
      useCache: useCache,
    );

    // Residual add
    hiddenStates = residual + hiddenStates;

    // Residual connection
    residual = hiddenStates;

    // 3. Post Attention Norm
    hiddenStates = postAttentionLayernorm.forward(
      hiddenStates,
      context: context,
    );

    // 4. MLP
    hiddenStates = mlp.forward(hiddenStates, context: context);

    // Residual add
    hiddenStates = residual + hiddenStates;

    return hiddenStates;
  }

  @override
  Iterable<Module> get submodules => [
    selfAttn,
    mlp,
    inputLayernorm,
    postAttentionLayernorm,
  ];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    selfAttn.resetParameters();
    mlp.resetParameters();
    inputLayernorm.resetParameters();
    postAttentionLayernorm.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {'layerIdx': layerIdx};
}
