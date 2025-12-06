/// Configuration class for S3-DiT (Scalable Single-Stream Diffusion Transformer)
class S3DiTConfig {
  /// Hidden size / embedding dimension
  final int hiddenSize;

  /// Number of attention heads
  final int numAttentionHeads;

  /// Number of transformer layers
  final int numHiddenLayers;

  /// Intermediate size multiplier for FFN (typically 2.5x hidden size)
  final double intermediateMultiplier;

  /// Dropout probability for attention
  final double attentionDropout;

  /// Dropout probability for hidden layers
  final double hiddenDropout;

  /// Epsilon for RMSNorm
  final double layerNormEps;

  /// Maximum sequence length
  final int maxPositionEmbeddings;

  /// Activation function for FFN ('silu' or 'gelu')
  final String hiddenActivation;

  /// Whether to use cosine QK normalization in attention
  final bool useQkNorm;

  /// Whether to use 3D RoPE (rotary position embedding)
  final bool use3dRope;

  /// Base frequency for RoPE
  final double ropeTheta;

  /// Number of key-value heads (for grouped-query attention, if different from num_attention_heads)
  final int? numKeyValueHeads;

  const S3DiTConfig({
    this.hiddenSize = 3072,
    this.numAttentionHeads = 24,
    this.numHiddenLayers = 28,
    this.intermediateMultiplier = 2.5,
    this.attentionDropout = 0.0,
    this.hiddenDropout = 0.0,
    this.layerNormEps = 1e-6,
    this.maxPositionEmbeddings = 8192,
    this.hiddenActivation = 'silu',
    this.useQkNorm = true,
    this.use3dRope = true,
    this.ropeTheta = 10000.0,
    this.numKeyValueHeads,
  });

  int get headDim => hiddenSize ~/ numAttentionHeads;

  int get intermediateSize => (hiddenSize * intermediateMultiplier).round();

  int get effectiveNumKeyValueHeads => numKeyValueHeads ?? numAttentionHeads;

  /// Default configuration for Z-Image Base (6B parameters)
  static const S3DiTConfig zImageBase = S3DiTConfig(
    hiddenSize: 3072,
    numAttentionHeads: 24,
    numHiddenLayers: 28,
    intermediateMultiplier: 2.5,
    attentionDropout: 0.0,
    hiddenDropout: 0.0,
    layerNormEps: 1e-6,
    maxPositionEmbeddings: 8192,
    hiddenActivation: 'silu',
    useQkNorm: true,
    use3dRope: true,
    ropeTheta: 10000.0,
  );

  /// Configuration for Z-Image Turbo (distilled version)
  static const S3DiTConfig zImageTurbo = S3DiTConfig(
    hiddenSize: 3072,
    numAttentionHeads: 24,
    numHiddenLayers: 28,
    intermediateMultiplier: 2.5,
    attentionDropout: 0.0,
    hiddenDropout: 0.0,
    layerNormEps: 1e-6,
    maxPositionEmbeddings: 8192,
    hiddenActivation: 'silu',
    useQkNorm: true,
    use3dRope: true,
    ropeTheta: 10000.0,
  );
}
