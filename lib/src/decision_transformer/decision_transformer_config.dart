import '../gpt2/gpt2_config.dart';

class DecisionTransformerConfig extends GPT2Config {
  final int stateDim;
  final int actDim;
  final int maxEpLen;

  DecisionTransformerConfig({
    this.stateDim = 17,
    this.actDim = 4,
    this.maxEpLen = 4096,
    super.vocabSize = 50257,
    super.nPositions = 1024,
    super.nEmbd = 768,
    super.nLayer = 12,
    super.nHead = 12,
    super.nInner = 0,
    super.activationFunction = 0.0,
    super.residPdrop = 0.1,
    super.embdPdrop = 0.1,
    super.attnPdrop = 0.1,
    super.layerNormEpsilon = 1e-5,
    super.scaleAttnWeights = true,
    super.scaleAttnByInverseLayerIdx = false,
    super.reorderAndUpcastAttn = false,
    super.useCache = true,
  });

  factory DecisionTransformerConfig.fromJson(Map<String, dynamic> json) {
    return DecisionTransformerConfig(
      stateDim: json['state_dim'] ?? 17,
      actDim: json['act_dim'] ?? 4,
      maxEpLen: json['max_ep_len'] ?? 4096,
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
