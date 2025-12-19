class DeepSeekConfig {
  final int vocabSize;
  final int hiddenSize;
  final int intermediateSize;
  final int numHiddenLayers;
  final int numAttentionHeads;
  final int numKeyValueHeads;
  final int nRoutedExperts;
  final int nSharedExperts;
  final int topK;
  final int moeIntermediateSize;
  final int qLoraRank;
  final int kvLoraRank;
  final int nopeHeadDim;
  final int ropeHeadDim;
  final double rotaryEmbBase;
  final double rmsNormEps;
  final int maxPositionEmbeddings;
  final bool useCache;
  final bool tieWordEmbeddings;
  final double attentionDropout;
  final String hiddenAct;

  // MLA specific derived properties
  int get headDim => nopeHeadDim + ropeHeadDim;

  DeepSeekConfig({
    this.vocabSize = 102400,
    this.hiddenSize = 4096,
    this.intermediateSize = 11008,
    this.numHiddenLayers = 30,
    this.numAttentionHeads = 32,
    this.numKeyValueHeads = 32,
    this.nRoutedExperts = 64,
    this.nSharedExperts = 2,
    this.topK = 6,
    this.moeIntermediateSize = 1408,
    this.qLoraRank = 1536,
    this.kvLoraRank = 512,
    this.nopeHeadDim = 128,
    this.ropeHeadDim = 64,
    this.rotaryEmbBase = 10000.0,
    this.rmsNormEps = 1e-6,
    this.maxPositionEmbeddings = 4096,
    this.useCache = true,
    this.tieWordEmbeddings = false,
    this.attentionDropout = 0.0,
    this.hiddenAct = "silu",
  });

  factory DeepSeekConfig.fromJson(Map<String, dynamic> json) {
    return DeepSeekConfig(
      vocabSize: json['vocab_size'] ?? 102400,
      hiddenSize: json['hidden_size'] ?? 4096,
      intermediateSize: json['intermediate_size'] ?? 11008,
      numHiddenLayers: json['num_hidden_layers'] ?? 30,
      numAttentionHeads: json['num_attention_heads'] ?? 32,
      numKeyValueHeads: json['num_key_value_heads'] ?? 32,
      nRoutedExperts: json['n_routed_experts'] ?? 64,
      nSharedExperts: json['n_shared_experts'] ?? 2,
      topK:
          json['num_experts_per_tok'] ??
          6, // Often called num_experts_per_tok in configs
      moeIntermediateSize: json['moe_intermediate_size'] ?? 1408,
      qLoraRank: json['q_lora_rank'] ?? 1536,
      kvLoraRank: json['kv_lora_rank'] ?? 512,
      nopeHeadDim: json['nope_head_dim'] ?? 128,
      ropeHeadDim: json['rope_head_dim'] ?? 64,
      rotaryEmbBase: (json['rope_theta'] ?? 10000.0).toDouble(),
      rmsNormEps: (json['rms_norm_eps'] ?? 1e-6).toDouble(),
      maxPositionEmbeddings: json['max_position_embeddings'] ?? 4096,
      useCache: json['use_cache'] ?? true,
      tieWordEmbeddings: json['tie_word_embeddings'] ?? false,
      attentionDropout: (json['attention_dropout'] ?? 0.0).toDouble(),
      hiddenAct: json['hidden_act'] ?? "silu",
    );
  }
}
