import 'package:kamma/kamma.dart';
import 'package:kamma/src/gemma/gemma_config.dart';
import 'package:kamma/src/gemma/gemma_decoder_layer.dart';
import 'package:kamma/src/gemma/gemma_rms_norm.dart';
import 'package:kamma/src/gemma/gemma_rotary_embedding.dart';
import 'package:tensor/tensor.dart';
import 'dart:math' as math;

class GemmaModel extends Module {
  final EmbeddingLayer embedTokens;
  final List<GemmaDecoderLayer> layers;
  final GemmaRMSNorm norm;
  final GemmaRotaryEmbedding rotaryEmb;
  final int hiddenSize;
  final bool isCausal;

  GemmaModel({
    required super.name,
    required this.embedTokens,
    required this.layers,
    required this.norm,
    required this.rotaryEmb,
    required this.hiddenSize,
    required this.isCausal,
  });

  static GemmaModel make({
    required GemmaConfig config,
    required Activation activation,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) {
    final headDim =
        config.headDim; // or config.hiddenSize ~/ config.numAttentionHeads;

    // Check consistency
    // GemmaConfig has headDim explicitly.

    return GemmaModel(
      name: 'model',
      hiddenSize: config.hiddenSize,
      isCausal: true,
      embedTokens: EmbeddingLayer.make(
        numEmbeddings: config.vocabSize,
        embedDim: config.hiddenSize,
        name: 'embed_tokens',
        paddingIdx: config.padTokenId,
      ),
      rotaryEmb: GemmaRotaryEmbedding.make(
        dim: headDim,
        base: config.ropeTheta,
        maxPositionEmbeddings: config.maxPositionEmbeddings,
      ),
      layers: List.generate(
        config.numHiddenLayers,
        (i) => GemmaDecoderLayer.make(
          layerIdx: i,
          hiddenSize: config.hiddenSize,
          intermediateSize: config.intermediateSize,
          numHeads: config.numAttentionHeads,
          numKeyValueHeads: config.numKeyValueHeads,
          headDim: headDim,
          maxPositionEmbeddings: config.maxPositionEmbeddings,
          ropeTheta: config.ropeTheta,
          rmsNormEps: config.rmsNormEps,
          activation: activation,
          attentionBias: config.attentionBias,
          attentionDropoutProb: config.attentionDropout,
          isCausal: true,
          attentionMethod: attentionMethod,
        ),
      ),
      norm: GemmaRMSNorm.make(
        name: 'norm',
        normalizedShape: [config.hiddenSize],
        eps: config.rmsNormEps,
      ),
    );
  }

  static Future<GemmaModel> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required GemmaConfig config,
    required Activation activation,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) async {
    final embedTokens = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      prefix: 'model.embed_tokens.',
      name: 'embed_tokens',
      paddingIdx: config.padTokenId,
    );

    final layers = <GemmaDecoderLayer>[];
    for (int i = 0; i < config.numHiddenLayers; i++) {
      layers.add(
        await GemmaDecoderLayer.loadFromSafeTensor(
          loader,
          layerIdx: i,
          prefix: 'model.layers.$i.',
          hiddenSize: config.hiddenSize,
          numHeads: config.numAttentionHeads,
          numKeyValueHeads: config.numKeyValueHeads,
          headDim: config.headDim,
          activation: activation,
          rmsNormEps: config.rmsNormEps,
          ropeTheta: config.ropeTheta,
          maxPositionEmbeddings: config.maxPositionEmbeddings,
          attentionDropoutProb: config.attentionDropout,
          isCausal: true,
          attentionBias: config.attentionBias,
          attentionMethod: attentionMethod,
        ),
      );
    }

    final norm = await GemmaRMSNorm.loadFromSafeTensor(
      loader,
      prefix: 'model.norm.',
      name: 'norm',
      normalizedShape: [config.hiddenSize],
      eps: config.rmsNormEps,
    );

    return GemmaModel(
      name: 'model',
      embedTokens: embedTokens,
      layers: layers,
      norm: norm,
      rotaryEmb: GemmaRotaryEmbedding.make(
        dim: config.headDim,
        base: config.ropeTheta,
        maxPositionEmbeddings: config.maxPositionEmbeddings,
      ),
      hiddenSize: config.hiddenSize,
      isCausal: true,
    );
  }

  ({Tensor hiddenStates, List<Tensor>? allHiddenStates}) forward(
    Tensor inputIds, {
    required Context context,
    Tensor? inputEmbeds,
    Tensor? attentionMask,
    Tensor? positionIds,
    bool useCache = false,
    bool returnHiddenStates = false,
  }) {
    context.onloadModule(this);

    Tensor hiddenStates;
    if (inputEmbeds != null) {
      hiddenStates = inputEmbeds;
    } else {
      hiddenStates = embedTokens.forward(inputIds, context: context);
      // Gemma specifically scales embeddings by sqrt(dim)
      hiddenStates = hiddenStates * math.sqrt(hiddenSize);
    }

    List<Tensor>? allHiddenStates;
    if (returnHiddenStates) {
      allHiddenStates = [hiddenStates];
    }

    // Prepare RoPE position embeddings
    // We compute cos/sin here once for all layers
    if (positionIds == null) {
      final seqLen = inputIds.shape[1]; // Or embedding shape
      final device = hiddenStates.device;
      positionIds = Tensor.arange(0, seqLen, device: device).unsqueeze(0);
      // If we are using cache (inference), we need to account for past length?
      // Usually caller handles positionIds for inference with cache.
      // If useCache is true, positionIds should be provided or inferred from cache length?
      // For now assume basic behavior.
    }

    // Causal Mask
    if (attentionMask == null && isCausal) {
      // Assuming causal
      // If simple inference, maybe we rely on attention layer to generate mask?
      // GemmaDecoderLayer passes attentionMask to Attention.
      // If null, Attention (if SDPA) usually assumes causal if configured.
      // But if we want explicit mask:

      // Let's create a basic causal mask if needed, but usually AttentionMethod handles it if null.
      // We pass null to let AttentionMethod handle it.
    }

    final (:cos, :sin) = rotaryEmb.forward(positionIds, context: context);
    final positionEmbeddings = (cos: cos, sin: sin);

    for (final layer in layers) {
      hiddenStates = layer.forward(
        hiddenStates,
        context: context,
        attentionMask: attentionMask,
        positionIds: positionIds,
        positionEmbeddings: positionEmbeddings,
        useCache: useCache,
      );
      if (returnHiddenStates) {
        allHiddenStates!.add(hiddenStates);
      }
    }

    hiddenStates = norm.forward(hiddenStates, context: context);

    return (hiddenStates: hiddenStates, allHiddenStates: allHiddenStates);
  }

  void resetKeyValueCache() {
    for (final layer in layers) {
      layer.resetKeyValueCache();
    }
  }

  @override
  Iterable<Module> get submodules => [embedTokens, ...layers, norm];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    embedTokens.resetParameters();
    for (final layer in layers) layer.resetParameters();
    norm.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {};
}
