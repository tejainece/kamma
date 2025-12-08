class GptOssConfig {
  final int vocabSize;
  final int nPositions;
  final int embedDim;
  final int nLayer;
  final int nHead;
  final int nInner;
  final double activationFunction; // Placeholder
  final double residPdrop;
  final double embdPdrop;
  final double attnPdrop;
  final double layerNormEpsilon;
  final bool scaleAttnWeights;
  final bool scaleAttnByInverseLayerIdx;
  final bool reorderAndUpcastAttn;
  final bool useCache;

  // New params for GPT-OSS
  final int numExperts;
  final int numExpertsPerToken;
  final double ropeTheta;
  final double? ropeScaling;
  final int? numKeyValueHeads;
  final double rmsNormEps;

  GptOssConfig({
    this.vocabSize = 50257,
    this.nPositions = 1024,
    this.embedDim = 768,
    this.nLayer = 12,
    this.nHead = 12,
    this.nInner = 0,
    this.activationFunction = 0.0,
    this.residPdrop = 0.1,
    this.embdPdrop = 0.1,
    this.attnPdrop = 0.1,
    this.layerNormEpsilon = 1e-5,
    this.scaleAttnWeights = true,
    this.scaleAttnByInverseLayerIdx = false,
    this.reorderAndUpcastAttn = false,
    this.useCache = true,
    this.numExperts = 32,
    this.numExpertsPerToken = 4,
    this.ropeTheta = 10000.0,
    this.ropeScaling,
    this.numKeyValueHeads,
    this.rmsNormEps = 1e-6,
  });

  factory GptOssConfig.fromJson(Map<String, dynamic> json) {
    return GptOssConfig(
      vocabSize: json['vocab_size'] ?? 50257,
      nPositions:
          json['n_positions'] ?? json['max_position_embeddings'] ?? 1024,
      embedDim: json['n_embd'] ?? json['hidden_size'] ?? 768,
      nLayer: json['n_layer'] ?? json['num_hidden_layers'] ?? 12,
      nHead: json['n_head'] ?? json['num_attention_heads'] ?? 12,
      nInner: json['n_inner'] ?? json['intermediate_size'] ?? 0,
      residPdrop: (json['resid_pdrop'] ?? 0.1).toDouble(),
      embdPdrop: (json['embd_pdrop'] ?? 0.1).toDouble(),
      attnPdrop: (json['attn_pdrop'] ?? json['attention_dropout'] ?? 0.1)
          .toDouble(),
      layerNormEpsilon: (json['layer_norm_epsilon'] ?? 1e-5).toDouble(),
      scaleAttnWeights: json['scale_attn_weights'] ?? true,
      scaleAttnByInverseLayerIdx:
          json['scale_attn_by_inverse_layer_idx'] ?? false,
      reorderAndUpcastAttn: json['reorder_and_upcast_attn'] ?? false,
      useCache: json['use_cache'] ?? true,
      numExperts: json['num_experts'] ?? json['num_local_experts'] ?? 32,
      numExpertsPerToken:
          json['num_experts_per_token'] ?? json['num_experts_per_tok'] ?? 4,
      ropeTheta: (json['rope_theta'] ?? 10000.0).toDouble(),
      ropeScaling: json['rope_scaling'] is Map
          ? (json['rope_scaling']['factor'] as num?)?.toDouble()
          : (json['rope_scaling'] as num?)?.toDouble(),
      numKeyValueHeads: json['num_key_value_heads'],
      rmsNormEps: (json['rms_norm_eps'] ?? 1e-6).toDouble(),
    );
  }
}
