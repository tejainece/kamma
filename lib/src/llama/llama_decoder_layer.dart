import 'package:kamma/kamma.dart';

import 'llama_rms_norm.dart';

class LlamaDecoderLayer extends Module implements SimpleModule {
  final int layerIdx;

  final LlamaAttention selfAttn;
  final LlamaMLP mlp;
  final LlamaRMSNorm inputLayernorm;
  final LlamaRMSNorm postAttentionLayernorm;

  LlamaDecoderLayer(
    this.layerIdx, {
    required this.selfAttn,
    required this.mlp,
    required this.inputLayernorm,
    required this.postAttentionLayernorm,
  }) : super(name: 'layers.$layerIdx');

  @override
  Tensor forward(
    Tensor embeddings, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    ({Tensor cos, Tensor sin})? positionEmbeddings,
    bool useCache = false,
  }) {
    context.onloadModule(this);

    Tensor residual = embeddings;
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

  static LlamaDecoderLayer make({
    required int layerIdx,
    required int hiddenSize,
    required int intermediateSize,
    required int numHeads,
    required int numKeyValueHeads,
    required int headDim,
    required int maxPositionEmbeddings,
    required double ropeTheta,
    required double rmsNormEps,
    required Activation activation,
    required bool hasAttentionBias,
    required double attentionDropoutProb,
    required bool isCausal,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) {
    return LlamaDecoderLayer(
      layerIdx,
      selfAttn: LlamaAttention.make(
        name: 'self_attn',
        layerIdx: layerIdx,
        numHeads: numHeads,
        embedDim: hiddenSize,
        maxPositionEmbeddings: maxPositionEmbeddings,
        ropeTheta: ropeTheta,
        hasAttentionBias: hasAttentionBias,
        numKeyValueHeads: numKeyValueHeads,
        attentionDropoutProb: attentionDropoutProb,
        isCausal: isCausal,
        attentionMethod: attentionMethod,
      ),
      mlp: LlamaMLP.make(
        embedDim: hiddenSize,
        intermediateSize: intermediateSize,
        activation: activation,
        hasBias: false, // TODO Assuming default TODO check config.mlpBias usage
      ),
      inputLayernorm: LlamaRMSNorm.make(
        name: 'input_layernorm',
        dim: hiddenSize,
        eps: rmsNormEps,
      ),
      postAttentionLayernorm: LlamaRMSNorm.make(
        name: 'post_attention_layernorm',
        dim: hiddenSize,
        eps: rmsNormEps,
      ),
    );
  }

  static Future<LlamaDecoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required int layerIdx,
    required String prefix,
    // TODO cant we compute this from attention
    required int embedDim,
    required int numHeads,
    required double attentionDropoutProb,
    required int maxPositionEmbeddings,
    required double ropeTheta,
    required bool isCausal,
    required double rmsNormEps,
    required Activation activation,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) async {
    final selfAttn = await LlamaAttention.loadFromSafeTensor(
      loader,
      prefix: '${prefix}self_attn.',
      name: 'self_attn',
      layerIdx: layerIdx,
      numHeads: numHeads,
      attentionDropoutProb: attentionDropoutProb,
      maxPositionEmbeddings: maxPositionEmbeddings,
      ropeTheta: ropeTheta,
      isCausal: isCausal,
      attentionMethod: attentionMethod,
    );
    return LlamaDecoderLayer(
      layerIdx,
      selfAttn: selfAttn,
      mlp: await LlamaMLP.loadFromSafeTensor(
        loader,
        prefix: '${prefix}mlp.',
        activation: activation,
      ),
      inputLayernorm: await LlamaRMSNorm.loadFromSafeTensor(
        loader,
        prefix: '${prefix}input_layernorm.',
        name: 'input_layernorm',
        normalizedShape: selfAttn.embedDim,
        eps: rmsNormEps,
      ),
      postAttentionLayernorm: await LlamaRMSNorm.loadFromSafeTensor(
        loader,
        prefix: '${prefix}post_attention_layernorm.',
        name: 'post_attention_layernorm',
        normalizedShape: selfAttn.embedDim,
        eps: rmsNormEps,
      ),
    );
  }

  void resetKeyValueCache() {
    selfAttn.resetKeyValueCache();
  }
}
