class GPT2Config {
  final int vocabSize;
  final int nPositions;
  final int nEmbd;
  final int nLayer;
  final int nHead;
  final int nInner;
  final double
  activationFunction; // Not used directly, but kept for compatibility
  final double residPdrop;
  final double embdPdrop;
  final double attnPdrop;
  final double layerNormEpsilon;
  final bool scaleAttnWeights;
  final bool scaleAttnByInverseLayerIdx;
  final bool reorderAndUpcastAttn;
  final bool useCache;

  GPT2Config({
    this.vocabSize = 50257,
    this.nPositions = 1024,
    this.nEmbd = 768,
    this.nLayer = 12,
    this.nHead = 12,
    this.nInner = 0, // 0 means nEmbd * 4
    this.activationFunction = 0.0, // Placeholder
    this.residPdrop = 0.1,
    this.embdPdrop = 0.1,
    this.attnPdrop = 0.1,
    this.layerNormEpsilon = 1e-5,
    this.scaleAttnWeights = true,
    this.scaleAttnByInverseLayerIdx = false,
    this.reorderAndUpcastAttn = false,
    this.useCache = true,
  });

  factory GPT2Config.fromJson(Map<String, dynamic> json) {
    return GPT2Config(
      vocabSize: json['vocab_size'] ?? 50257,
      nPositions: json['n_positions'] ?? 1024,
      nEmbd: json['n_embd'] ?? 768,
      nLayer: json['n_layer'] ?? 12,
      nHead: json['n_head'] ?? 12,
      nInner: json['n_inner'] ?? 0,
      residPdrop: (json['resid_pdrop'] ?? 0.1).toDouble(),
      embdPdrop: (json['embd_pdrop'] ?? 0.1).toDouble(),
      attnPdrop: (json['attn_pdrop'] ?? 0.1).toDouble(),
      layerNormEpsilon: (json['layer_norm_epsilon'] ?? 1e-5).toDouble(),
      scaleAttnWeights: json['scale_attn_weights'] ?? true,
      scaleAttnByInverseLayerIdx:
          json['scale_attn_by_inverse_layer_idx'] ?? false,
      reorderAndUpcastAttn: json['reorder_and_upcast_attn'] ?? false,
      useCache: json['use_cache'] ?? true,
    );
  }
}
