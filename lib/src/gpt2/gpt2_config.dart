class GPT2Config {
  final int vocabSize;
  final int nPositions;
  final int embedDim;

  /// Number of attention layers serially connected to enrich the context-aware embeddings
  final int numLayers;

  /// Number of heads in attention
  final int numHeads;

  /// Number of inner dimensions in the feed-forward network
  final int? mlpInnerDim;
  final String activationFunction;
  final double residualDropoutProbability;
  final double embedDropoutProbability;
  final double attentionDropoutProbability;
  final double layerNormEpsilon;
  final bool scaleAttnWeights;
  final bool scaleAttnByInverseLayerIdx;
  final bool reorderAndUpcastAttn;
  final bool useCache;
  final int maxPositionEmbeddings;

  GPT2Config({
    this.vocabSize = 50257,
    this.nPositions = 1024,
    this.embedDim = 768,
    this.numLayers = 12,
    this.numHeads = 12,
    this.mlpInnerDim,
    this.activationFunction = "gelu_new",
    this.residualDropoutProbability = 0.1,
    this.embedDropoutProbability = 0.1,
    this.attentionDropoutProbability = 0.1,
    this.layerNormEpsilon = 1e-5,
    this.scaleAttnWeights = true,
    this.scaleAttnByInverseLayerIdx = false,
    this.reorderAndUpcastAttn = false,
    this.useCache = true,
    this.maxPositionEmbeddings = 1024,
  });

  factory GPT2Config.fromJson(Map<String, dynamic> json) {
    return GPT2Config(
      vocabSize: json['vocab_size'] ?? 50257,
      nPositions: json['n_positions'] ?? 1024,
      embedDim: json['n_embd'] ?? 768,
      numLayers: json['n_layer'] ?? 12,
      numHeads: json['n_head'] ?? 12,
      mlpInnerDim: json['n_inner'],
      activationFunction: json['activation_function'] ?? "gelu_new",
      residualDropoutProbability: (json['resid_pdrop'] ?? 0.1).toDouble(),
      embedDropoutProbability: (json['embd_pdrop'] ?? 0.1).toDouble(),
      attentionDropoutProbability: (json['attn_pdrop'] ?? 0.1).toDouble(),
      layerNormEpsilon: (json['layer_norm_epsilon'] ?? 1e-5).toDouble(),
      scaleAttnWeights: json['scale_attn_weights'] ?? true,
      scaleAttnByInverseLayerIdx:
          json['scale_attn_by_inverse_layer_idx'] ?? false,
      reorderAndUpcastAttn: json['reorder_and_upcast_attn'] ?? false,
      useCache: json['use_cache'] ?? true,
    );
  }
}
