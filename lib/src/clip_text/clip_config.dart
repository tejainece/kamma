import 'package:tensor/src/nn/activation.dart';

class ClipTextConfig {
  final int vocabSize;
  final int intermediateSize;
  final int projectionDim;
  final int numHiddenLayers;
  final int numAttentionHeads;

  /// TODO what does this do?
  final int maxPositionEmbeddings;

  /// Embed dim/hidden size
  /// Number of learned
  final int embedDim;
  final Activation activation;
  final double layerNormEps;
  final String? padWith;
  final double attentionDropout;

  const ClipTextConfig({
    required this.vocabSize,
    required this.intermediateSize,
    required this.projectionDim,
    required this.numHiddenLayers,
    required this.numAttentionHeads,
    required this.maxPositionEmbeddings,
    required this.embedDim,
    required this.activation,
    required this.layerNormEps,
    required this.padWith,
    required this.attentionDropout,
  });

  static const ClipTextConfig v1_5 = ClipTextConfig(
    vocabSize: 49408,
    embedDim: 768,
    activation: Activation.quickGelu,
    intermediateSize: 3072,
    maxPositionEmbeddings: 77,
    padWith: null,
    numHiddenLayers: 12,
    numAttentionHeads: 12,
    projectionDim: 768,
    layerNormEps: 1e-5,
    attentionDropout: 0,
  );

  static const ClipTextConfig v2_1 = ClipTextConfig(
    vocabSize: 49408,
    embedDim: 1024,
    activation: Activation.gelu,
    intermediateSize: 4096,
    maxPositionEmbeddings: 77,
    padWith: "!",
    numHiddenLayers: 23,
    numAttentionHeads: 16,
    projectionDim: 512,
    layerNormEps: 1e-5,
    attentionDropout: 0,
  );
}
