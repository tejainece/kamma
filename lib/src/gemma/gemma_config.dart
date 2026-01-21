class GemmaConfig {
  final int vocabSize;
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int headDim;
  final String hiddenAct;
  final int maxPositionEmbeddings;
  final double initializerRange;
  final double rmsNormEps;
  final bool useCache;
  final int? padTokenId;
  final int bosTokenId;
  final int eosTokenId;
  final double ropeTheta;
  final bool attentionBias;
  final double attentionDropout;

  GemmaConfig({
    this.vocabSize = 256000,
    this.hiddenSize = 3072,
    this.intermediateSize = 24576,
    this.numHiddenLayers = 28,
    this.numAttentionHeads = 16,
    this.numKeyValueHeads = 16,
    this.headDim = 256,
    this.hiddenAct = "gelu",
    this.maxPositionEmbeddings = 8192,
    this.initializerRange = 0.02,
    this.rmsNormEps = 1e-6,
    this.useCache = true,
    this.padTokenId,
    this.bosTokenId = 2,
    this.eosTokenId = 1,
    this.ropeTheta = 10000.0,
    this.attentionBias = false,
    this.attentionDropout = 0.0,
  });

  factory GemmaConfig.fromJson(Map<String, dynamic> json) {
    return GemmaConfig(
      vocabSize: json['vocab_size'] ?? 256000,
      hiddenSize: json['hidden_size'] ?? 3072,
      intermediateSize: json['intermediate_size'] ?? 24576,
      numHiddenLayers: json['num_hidden_layers'] ?? 28,
      numAttentionHeads: json['num_attention_heads'] ?? 16,
      numKeyValueHeads: json['num_key_value_heads'] ?? 16,
      headDim: json['head_dim'] ?? 256,
      hiddenAct: json['hidden_act'] ?? "gelu",
      maxPositionEmbeddings: json['max_position_embeddings'] ?? 8192,
      initializerRange: (json['initializer_range'] ?? 0.02).toDouble(),
      rmsNormEps: (json['rms_norm_eps'] ?? 1e-6).toDouble(),
      useCache: json['use_cache'] ?? true,
      padTokenId: json['pad_token_id'],
      bosTokenId: json['bos_token_id'] ?? 2,
      eosTokenId: json['eos_token_id'] ?? 1,
      ropeTheta: (json['rope_theta'] ?? 10000.0).toDouble(),
      attentionBias: json['attention_bias'] ?? false,
      attentionDropout: (json['attention_dropout'] ?? 0.0).toDouble(),
    );
  }
}
