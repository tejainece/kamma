class LlamaConfig {
  final int vocabSize;
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final String hiddenAct;
  final int maxPositionEmbeddings;
  final double initializerRange;
  final double rmsNormEps;
  final bool useCache;
  final int? padTokenId;
  final int bosTokenId;
  final int eosTokenId;
  final bool tieWordEmbeddings;
  final double ropeTheta;
  final Map<String, dynamic>? ropeScaling;
  final bool attentionBias;
  final double attentionDropout;
  final bool mlpBias;
  final int headDim;
  // Pretraining TP not usually needed for inference, but keeping for completeness
  final int pretrainingTp;

  LlamaConfig({
    this.vocabSize = 32000,
    this.hiddenSize = 4096,
    this.intermediateSize = 11008,
    this.numHiddenLayers = 32,
    this.numAttentionHeads = 32,
    int? numKeyValueHeads,
    this.hiddenAct = "silu",
    this.maxPositionEmbeddings = 2048,
    this.initializerRange = 0.02,
    this.rmsNormEps = 1e-6,
    this.useCache = true,
    this.padTokenId,
    this.bosTokenId = 1,
    this.eosTokenId = 2,
    this.pretrainingTp = 1,
    this.tieWordEmbeddings = false,
    this.ropeTheta = 10000.0,
    this.ropeScaling,
    this.attentionBias = false,
    this.attentionDropout = 0.0,
    this.mlpBias = false,
    int? headDim,
  }) : this.numKeyValueHeads = numKeyValueHeads ?? numAttentionHeads,
       this.headDim = headDim ?? (hiddenSize ~/ numAttentionHeads);

  factory LlamaConfig.fromJson(Map<String, dynamic> json) {
    return LlamaConfig(
      vocabSize: json['vocab_size'] ?? 32000,
      hiddenSize: json['hidden_size'] ?? 4096,
      intermediateSize: json['intermediate_size'] ?? 11008,
      numHiddenLayers: json['num_hidden_layers'] ?? 32,
      numAttentionHeads: json['num_attention_heads'] ?? 32,
      numKeyValueHeads: json['num_key_value_heads'],
      hiddenAct: json['hidden_act'] ?? "silu",
      maxPositionEmbeddings: json['max_position_embeddings'] ?? 2048,
      initializerRange: (json['initializer_range'] ?? 0.02).toDouble(),
      rmsNormEps: (json['rms_norm_eps'] ?? 1e-6).toDouble(),
      useCache: json['use_cache'] ?? true,
      padTokenId: json['pad_token_id'],
      bosTokenId: json['bos_token_id'] ?? 1,
      eosTokenId: json['eos_token_id'] ?? 2,
      pretrainingTp: json['pretraining_tp'] ?? 1,
      tieWordEmbeddings: json['tie_word_embeddings'] ?? false,
      ropeTheta: (json['rope_theta'] ?? 10000.0).toDouble(),
      ropeScaling: json['rope_scaling'],
      attentionBias: json['attention_bias'] ?? false,
      attentionDropout: (json['attention_dropout'] ?? 0.0).toDouble(),
      mlpBias: json['mlp_bias'] ?? false,
      headDim: json['head_dim'],
    );
  }
}
