import 'package:kamma/kamma.dart';
import 'package:kamma/src/gemma/gemma_attention.dart';
import 'package:kamma/src/gemma/gemma_mlp.dart';
import 'package:kamma/src/gemma/gemma_rms_norm.dart';
import 'package:tensor/tensor.dart';

class GemmaDecoderLayer extends Module {
  final GemmaAttention attention;
  final GemmaMLP mlp;
  final GemmaRMSNorm inputLayerNorm;
  final GemmaRMSNorm postAttentionLayerNorm;

  GemmaDecoderLayer({
    required super.name,
    required this.attention,
    required this.mlp,
    required this.inputLayerNorm,
    required this.postAttentionLayerNorm,
  });

  static GemmaDecoderLayer make({
    required int layerIdx,
    required int hiddenSize,
    required int intermediateSize,
    required int numHeads,
    required int numKeyValueHeads,
    required int headDim,
    required int maxPositionEmbeddings,
    required double ropeTheta,
    required double rmsNormEps,
    required Activation activation,
    required bool attentionBias,
    required double attentionDropoutProb,
    required bool isCausal,
    required GPT2AttentionMethodType attentionMethod,
  }) {
    return GemmaDecoderLayer(
      name: 'layers.$layerIdx',
      attention: GemmaAttention.make(
        name: 'self_attn',
        layerIdx: layerIdx,
        numHeads: numHeads,
        numKeyValueHeads: numKeyValueHeads,
        headDim: headDim,
        embedDim: hiddenSize,
        isCausal: isCausal,
        ropeTheta: ropeTheta,
        maxPositionEmbeddings: maxPositionEmbeddings,
        attentionBias: attentionBias,
        attentionDropoutProb: attentionDropoutProb,
        attentionMethod: attentionMethod,
      ),
      mlp: GemmaMLP.make(
        name: 'mlp',
        hiddenSize: hiddenSize,
        intermediateSize: intermediateSize,
        activation: activation,
      ),
      inputLayerNorm: GemmaRMSNorm.make(
        name: 'input_layernorm',
        normalizedShape: [hiddenSize],
        eps: rmsNormEps,
      ),
      postAttentionLayerNorm: GemmaRMSNorm.make(
        name: 'post_attention_layernorm',
        normalizedShape: [hiddenSize],
        eps: rmsNormEps,
      ),
    );
  }

  static Future<GemmaDecoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required int layerIdx,
    required String prefix,
    required int hiddenSize,
    required int numHeads,
    required int numKeyValueHeads,
    required int headDim,
    required Activation activation,
    required double rmsNormEps,
    required double ropeTheta,
    required int maxPositionEmbeddings,
    required double attentionDropoutProb,
    required bool isCausal,
    required bool attentionBias,
    required GPT2AttentionMethodType attentionMethod,
  }) async {
    return GemmaDecoderLayer(
      name: 'layers.$layerIdx',
      attention: await GemmaAttention.loadFromSafeTensor(
        loader,
        prefix: '${prefix}self_attn.',
        name: 'self_attn',
        layerIdx: layerIdx,
        numHeads: numHeads,
        numKeyValueHeads: numKeyValueHeads,
        headDim: headDim,
        embedDim: hiddenSize,
        isCausal: isCausal,
        ropeTheta: ropeTheta,
        maxPositionEmbeddings: maxPositionEmbeddings,
        attentionDropoutProb: attentionDropoutProb,
        attentionMethod: attentionMethod,
      ),
      mlp: await GemmaMLP.loadFromSafeTensor(
        loader,
        prefix: '${prefix}mlp.',
        name: 'mlp',
        activation: activation,
      ),
      inputLayerNorm: await GemmaRMSNorm.loadFromSafeTensor(
        loader,
        prefix: '${prefix}input_layernorm.',
        name: 'input_layernorm',
        normalizedShape: [hiddenSize],
        eps: rmsNormEps,
      ),
      postAttentionLayerNorm: await GemmaRMSNorm.loadFromSafeTensor(
        loader,
        prefix: '${prefix}post_attention_layernorm.',
        name: 'post_attention_layernorm',
        normalizedShape: [hiddenSize],
        eps: rmsNormEps,
      ),
    );
  }

  Tensor forward(
    Tensor hiddenStates, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    ({Tensor cos, Tensor sin})? positionEmbeddings,
    bool useCache = false,
  }) {
    context.onloadModule(this);

    // Residual connection
    Tensor residual = hiddenStates;

    hiddenStates = inputLayerNorm.forward(hiddenStates, context: context);
    hiddenStates = attention.forward(
      hiddenStates,
      context: context,
      attentionMask: attentionMask,
      positionIds: positionIds,
      positionEmbeddings: positionEmbeddings,
      useCache: useCache,
    );
    hiddenStates = residual + hiddenStates;

    // Residual connection
    residual = hiddenStates;

    hiddenStates = postAttentionLayerNorm.forward(
      hiddenStates,
      context: context,
    );
    hiddenStates = mlp.forward(hiddenStates, context: context);
    hiddenStates = residual + hiddenStates;

    return hiddenStates;
  }

  void resetKeyValueCache() {
    attention.resetKeyValueCache();
  }

  @override
  Iterable<Module> get submodules => [
    attention,
    mlp,
    inputLayerNorm,
    postAttentionLayerNorm,
  ];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    attention.resetParameters();
    mlp.resetParameters();
    inputLayerNorm.resetParameters();
    postAttentionLayerNorm.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {};
}
