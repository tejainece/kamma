class Gemma3Config {
  final int vocabSize;
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final String hiddenActivation;
  final int maxPositionEmbeddings;
  final double initializerRange;
  final double rmsNormEps;
  final bool useCache;
  final int padTokenId;
  final int eosTokenId;
  final int bosTokenId;
  final bool tieWordEmbeddings;
  final double ropeTheta;
  final bool attentionBias;
  final double attentionDropout;
  final double queryPreAttnScalar;
  final int slidingWindow;
  final List<String>? layerTypes;
  final double? finalLogitSoftcapping;
  final double? attnLogitSoftcapping;
  final Map<String, dynamic>? ropeScaling;
  final double ropeLocalBaseFreq;
  final bool useBidirectionalAttention;

  Gemma3Config({
    this.vocabSize = 262208,
    this.hiddenSize = 2304,
    this.intermediateSize = 9216,
    this.numHiddenLayers = 26,
    this.numAttentionHeads = 8,
    this.numKeyValueHeads = 4,
    this.headDim = 256,
    this.hiddenActivation = "gelu_pytorch_tanh",
    this.maxPositionEmbeddings = 131072,
    this.initializerRange = 0.02,
    this.rmsNormEps = 1e-6,
    this.useCache = true,
    this.padTokenId = 0,
    this.eosTokenId = 1,
    this.bosTokenId = 2,
    this.tieWordEmbeddings = true,
    this.ropeTheta = 1000000.0,
    this.attentionBias = false,
    this.attentionDropout = 0.0,
    this.queryPreAttnScalar = 256,
    this.slidingWindow = 4096,
    this.layerTypes,
    this.finalLogitSoftcapping,
    this.attnLogitSoftcapping,
    this.ropeScaling,
    this.ropeLocalBaseFreq = 10000.0,
    this.useBidirectionalAttention = false,
  });

  factory Gemma3Config.fromJson(Map<String, dynamic> json) {
    return Gemma3Config(
      vocabSize: json['vocab_size'] ?? 262208,
      hiddenSize: json['hidden_size'] ?? 2304,
      intermediateSize: json['intermediate_size'] ?? 9216,
      numHiddenLayers: json['num_hidden_layers'] ?? 26,
      numAttentionHeads: json['num_attention_heads'] ?? 8,
      numKeyValueHeads: json['num_key_value_heads'] ?? 4,
      headDim: json['head_dim'] ?? 256,
      hiddenActivation: json['hidden_activation'] ?? "gelu_pytorch_tanh",
      maxPositionEmbeddings: json['max_position_embeddings'] ?? 131072,
      initializerRange: (json['initializer_range'] as num?)?.toDouble() ?? 0.02,
      rmsNormEps: (json['rms_norm_eps'] as num?)?.toDouble() ?? 1e-6,
      useCache: json['use_cache'] ?? true,
      padTokenId: json['pad_token_id'] ?? 0,
      eosTokenId: json['eos_token_id'] ?? 1,
      bosTokenId: json['bos_token_id'] ?? 2,
      tieWordEmbeddings: json['tie_word_embeddings'] ?? true,
      ropeTheta: (json['rope_theta'] as num?)?.toDouble() ?? 1000000.0,
      attentionBias: json['attention_bias'] ?? false,
      attentionDropout: (json['attention_dropout'] as num?)?.toDouble() ?? 0.0,
      queryPreAttnScalar:
          (json['query_pre_attn_scalar'] as num?)?.toDouble() ?? 256.0,
      slidingWindow: json['sliding_window'] ?? 4096,
      layerTypes: (json['layer_types'] as List?)?.cast<String>(),
      finalLogitSoftcapping: (json['final_logit_softcapping'] as num?)
          ?.toDouble(),
      attnLogitSoftcapping: (json['attn_logit_softcapping'] as num?)
          ?.toDouble(),
      ropeScaling: json['rope_scaling'],
      ropeLocalBaseFreq:
          (json['rope_local_base_freq'] as num?)?.toDouble() ?? 10000.0,
      useBidirectionalAttention: json['use_bidirectional_attention'] ?? false,
    );
  }
}
