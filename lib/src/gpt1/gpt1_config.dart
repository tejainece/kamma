class OpenAIGPTConfig {
  final int vocabSize;
  final int nPositions;
  final int nEmbd;
  final int nLayer;
  final int nHead;
  final String afn;
  final double residPdrop;
  final double embdPdrop;
  final double attnPdrop;
  final double layerNormEpsilon;
  final double initializerRange;
  // Summary parameters used in DoubleHeadsModel etc.
  final String summaryType;
  final bool summaryUseProj;
  final String? summaryActivation;
  final bool summaryProjToLabels;
  final double summaryFirstDropout;

  const OpenAIGPTConfig({
    this.vocabSize = 40478,
    this.nPositions = 512,
    this.nEmbd = 768,
    this.nLayer = 12,
    this.nHead = 12,
    this.afn = "gelu",
    this.residPdrop = 0.1,
    this.embdPdrop = 0.1,
    this.attnPdrop = 0.1,
    this.layerNormEpsilon = 1e-5,
    this.initializerRange = 0.02,
    this.summaryType = "cls_index",
    this.summaryUseProj = true,
    this.summaryActivation,
    this.summaryProjToLabels = true,
    this.summaryFirstDropout = 0.1,
  });

  factory OpenAIGPTConfig.fromJson(Map<String, dynamic> json) {
    return OpenAIGPTConfig(
      vocabSize: json['vocab_size'] ?? 40478,
      nPositions: json['n_positions'] ?? 512,
      nEmbd: json['n_embd'] ?? 768,
      nLayer: json['n_layer'] ?? 12,
      nHead: json['n_head'] ?? 12,
      afn: json['afn'] ?? "gelu",
      residPdrop: (json['resid_pdrop'] ?? 0.1).toDouble(),
      embdPdrop: (json['embd_pdrop'] ?? 0.1).toDouble(),
      attnPdrop: (json['attn_pdrop'] ?? 0.1).toDouble(),
      layerNormEpsilon: (json['layer_norm_epsilon'] ?? 1e-5).toDouble(),
      initializerRange: (json['initializer_range'] ?? 0.02).toDouble(),
      summaryType: json['summary_type'] ?? "cls_index",
      summaryUseProj: json['summary_use_proj'] ?? true,
      summaryActivation: json['summary_activation'],
      summaryProjToLabels: json['summary_proj_to_labels'] ?? true,
      summaryFirstDropout: (json['summary_first_dropout'] ?? 0.1).toDouble(),
    );
  }

  Map<String, dynamic> toJson() => {
    'vocab_size': vocabSize,
    'n_positions': nPositions,
    'n_embd': nEmbd,
    'n_layer': nLayer,
    'n_head': nHead,
    'afn': afn,
    'resid_pdrop': residPdrop,
    'embd_pdrop': embdPdrop,
    'attn_pdrop': attnPdrop,
    'layer_norm_epsilon': layerNormEpsilon,
    'initializer_range': initializerRange,
    'summary_type': summaryType,
    'summary_use_proj': summaryUseProj,
    'summary_activation': summaryActivation,
    'summary_proj_to_labels': summaryProjToLabels,
    'summary_first_dropout': summaryFirstDropout,
  };
}
