import 'package:kamma/kamma.dart';

/// S3-DiT (Scalable Single-Stream Diffusion Transformer) Model
///
/// Decoder-only transformer that processes unified multimodal sequences
/// (text tokens, VAE image tokens, semantic tokens) in a single stream.
class S3DiTModel extends Module {
  final S3DiTConfig config;
  final int hiddenSize;
  final int numLayers;

  final List<S3DiTBlock> layers;
  final RMSNorm finalNorm;

  S3DiTModel({
    required super.name,
    required this.config,
    required this.hiddenSize,
    required this.numLayers,
    required this.layers,
    required this.finalNorm,
  });

  /// Forward pass through the S3-DiT model
  ///
  /// Args:
  ///   hiddenStates: Input tensor [batch, seqLen, hiddenSize]
  ///   posX: X position indices for 3D RoPE [batch, seqLen] or [seqLen]
  ///   posY: Y position indices for 3D RoPE [batch, seqLen] or [seqLen]
  ///   posT: Temporal position indices for 3D RoPE [batch, seqLen] or [seqLen]
  ///   attentionMask: Optional attention mask [batch, 1, seqLen, seqLen]
  ///   context: Execution context
  ///
  /// Returns:
  ///   Output tensor [batch, seqLen, hiddenSize]
  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? posX,
    Tensor? posY,
    Tensor? posT,
    Tensor? attentionMask,
    required Context context,
  }) {
    context.onloadModule(this);

    // Pass through all transformer layers
    for (final layer in layers) {
      hiddenStates = layer.forward(
        hiddenStates,
        posX: posX,
        posY: posY,
        posT: posT,
        attentionMask: attentionMask,
        context: context,
      );
    }

    // Final normalization
    hiddenStates = finalNorm.forward(hiddenStates, context: context);

    return hiddenStates;
  }

  @override
  void resetParameters() {
    for (final layer in layers) {
      layer.resetParameters();
    }
    finalNorm.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...layers.expand((layer) => layer.parameters),
    ...finalNorm.parameters,
  ];

  @override
  late final Iterable<Module> submodules = [...layers, finalNorm];

  @override
  Map<String, dynamic> get meta => {
    'hiddenSize': hiddenSize,
    'numLayers': numLayers,
    'config': {
      'hiddenSize': config.hiddenSize,
      'numAttentionHeads': config.numAttentionHeads,
      'numHiddenLayers': config.numHiddenLayers,
      'intermediateSize': config.intermediateSize,
    },
  };

  /// Create S3-DiT model from configuration
  static S3DiTModel make({required S3DiTConfig config, String name = 's3dit'}) {
    final layers = <S3DiTBlock>[];

    for (int i = 0; i < config.numHiddenLayers; i++) {
      layers.add(S3DiTBlock.make(config: config, name: 'layers.$i'));
    }

    final finalNorm = RMSNorm.make(
      name: 'final_norm',
      normalizedShape: [config.hiddenSize],
      eps: config.layerNormEps,
    );

    return S3DiTModel(
      name: name,
      config: config,
      hiddenSize: config.hiddenSize,
      numLayers: config.numHiddenLayers,
      layers: layers,
      finalNorm: finalNorm,
    );
  }

  /// Load S3-DiT model from SafeTensor
  static Future<S3DiTModel> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required S3DiTConfig config,
    String prefix = '',
    String name = 's3dit',
  }) async {
    final layers = <S3DiTBlock>[];

    for (int i = 0; i < config.numHiddenLayers; i++) {
      // TODO: Implement layer loading from SafeTensor
      // This would require implementing loadFromSafeTensor for each component
      layers.add(S3DiTBlock.make(config: config, name: 'layers.$i'));
    }

    final finalNorm = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}final_norm.',
      name: 'final_norm',
      normalizedShape: [config.hiddenSize],
      eps: config.layerNormEps,
    );

    return S3DiTModel(
      name: name,
      config: config,
      hiddenSize: config.hiddenSize,
      numLayers: config.numHiddenLayers,
      layers: layers,
      finalNorm: finalNorm,
    );
  }
}
