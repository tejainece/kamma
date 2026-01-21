import 'package:kamma/kamma.dart';
import 'package:kamma/src/common/rope/rope.dart';

class LlamaModel extends Module {
  final EmbeddingLayer tokens;
  final List<LlamaDecoderLayer> layers;
  final RMSNorm norm;
  final LlamaRotaryEmbedding rotaryEmb;
  final bool isCausal;

  LlamaModel({
    required super.name,
    required this.tokens,
    required this.layers,
    required this.norm,
    required this.rotaryEmb,
    this.isCausal = true,
  });

  ({Tensor hiddenStates, List<Tensor>? allHiddenStates}) forward(
    Tensor inputIds, {
    required Context context,
    Tensor? inputEmbeds,
    Tensor? attentionMask,
    Tensor? positionIds,
    // TODO cache position
    bool useCache = false,
    bool returnHiddenStates = false,
  }) {
    context.onloadModule(this);

    Tensor hiddenStates = tokens.forward(inputIds, context: context);
    List<Tensor>? allHiddenStates;
    if (returnHiddenStates) {
      allHiddenStates = [hiddenStates];
    }

    // Prepare RoPE position embeddings
    if (positionIds == null) {
      final seqLen = inputIds.shape[1];
      final device = inputIds.device;
      positionIds = Tensor.arange(0, seqLen, device: device).unsqueeze(0);
    }

    if (attentionMask == null && isCausal) {
      final device = hiddenStates.device;
      final seqLen = inputIds.shape[1];

      // Generate on CPU to avoid MPS potential issues with tril/construction
      // causal_mask = (1 - tril(ones)) * -inf
      final ones = Tensor.ones([seqLen, seqLen]); // CPU
      final causalMaskCpu = (ones - ones.tril()) * -1e4;
      final causalMask = causalMaskCpu.to(device: device);

      // (B, 1, S, S)
      attentionMask = causalMask.unsqueeze(0).unsqueeze(0);
    }

    // Compute RoPE cos/sin once for all layers
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
    if (returnHiddenStates) {
      // Usually hidden states include the final one after norm too?
      // HF usually returns states BEFORE norm in 'hidden_states' tuple,
      // but let's check what testdata expects.
      // Testdata usually expects output of each layer block.
      // The last one is usually the output of the last block (before norm).
      // BUT the "final hidden state" is usually after norm.
      // Let's stick to adding output of each layer.
      // And maybe the final one after norm?
      // Let's check testdata keys. hidden_state_0 to hidden_state_15.
      // There are 16 layers. So hidden_state_0 is output of layer 0.
      // hidden_state_15 is output of layer 15.
      // The 'final' hidden state used for logits is after norm.
    }

    return (hiddenStates: hiddenStates, allHiddenStates: allHiddenStates);
  }

  @override
  Iterable<Module> get submodules => [tokens, ...layers, norm];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    tokens.resetParameters();
    for (final layer in layers) {
      layer.resetParameters();
    }
    norm.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {};

  static LlamaModel make({
    String name = 'model',
    required int numLayers,
    required int embedDim,
    required int numHeads,
    required int numKeyValueHeads,
    required int intermediateSize,
    required Activation activation,
    required double rmsNormEps,
    required double ropeTheta,
    required int maxPositionEmbeddings,
    required RopeArgs ropeArgs,
    required int vocabSize,
    required bool hasAttentionBias,
    required double attentionDropoutProb,
    int? padTokenId,
    required bool isCausal,
    required GPT2AttentionMethodType attentionMethod,
  }) {
    final int headDim = embedDim ~/ numHeads;

    final rotaryEmb = LlamaRotaryEmbedding.make(
      dim: headDim,
      base: ropeTheta,
      ropeArgs: ropeArgs,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );

    final tokens = EmbeddingLayer.make(
      numEmbeddings: vocabSize,
      embedDim: embedDim,
      name: 'embed_tokens',
      paddingIdx: padTokenId,
    );

    final layers = List.generate(numLayers, (i) {
      return LlamaDecoderLayer.make(
        layerIdx: i,
        hiddenSize: embedDim,
        intermediateSize: intermediateSize,
        numHeads: numHeads,
        numKeyValueHeads: numKeyValueHeads,
        headDim: headDim,
        maxPositionEmbeddings: maxPositionEmbeddings,
        ropeTheta: ropeTheta,
        rmsNormEps: rmsNormEps,
        activation: activation,
        hasAttentionBias: hasAttentionBias,
        attentionDropoutProb: attentionDropoutProb,
        isCausal: isCausal,
        attentionMethod: attentionMethod,
      );
    });

    final norm = RMSNorm([embedDim], eps: rmsNormEps);

    return LlamaModel(
      name: name,
      tokens: tokens,
      rotaryEmb: rotaryEmb,
      layers: layers,
      norm: norm,
      isCausal: isCausal,
    );
  }

  static Future<LlamaModel> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required int numLayers,
    required int embedDim,
    required int numHeads,
    required int numKeyValueHeads,
    required int intermediateSize,
    required Activation activation,
    required double rmsNormEps,
    required double ropeTheta,
    required int maxPositionEmbeddings,
    required RopeArgs ropeArgs,
    required int vocabSize,
    required double attentionDropoutProb,
    int? padTokenId,
    required bool isCausal,
    required GPT2AttentionMethodType attentionMethod,
  }) async {
    final tokens = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      prefix: 'model.embed_tokens.',
      name: 'embed_tokens',
      paddingIdx: padTokenId,
    );

    final layers = <LlamaDecoderLayer>[];
    for (int i = 0; i < numLayers; i++) {
      layers.add(
        await LlamaDecoderLayer.loadFromSafeTensor(
          loader,
          layerIdx: i,
          prefix: 'model.layers.$i.',
          embedDim: embedDim,
          numHeads: numHeads,
          activation: activation,
          rmsNormEps: rmsNormEps,
          ropeTheta: ropeTheta,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attentionDropoutProb: attentionDropoutProb,
          isCausal: isCausal,
          attentionMethod: attentionMethod,
        ),
      );
    }

    final int headDim = embedDim ~/ numHeads;

    final norm = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: 'model.norm.',
      name: 'norm',
      normalizedShape: [embedDim],
      eps: rmsNormEps,
    );

    return LlamaModel(
      name: 'model',
      tokens: tokens,
      layers: layers,
      norm: norm,
      rotaryEmb: LlamaRotaryEmbedding.make(
        dim: headDim,
        base: ropeTheta,
        ropeArgs: ropeArgs,
        maxPositionEmbeddings: maxPositionEmbeddings,
      ),
      isCausal: isCausal,
    );
  }

  void resetKeyValueCache() {
    for (final layer in layers) {
      layer.resetKeyValueCache();
    }
  }
}
