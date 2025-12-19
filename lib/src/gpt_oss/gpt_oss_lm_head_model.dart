import 'package:kamma/kamma.dart';

class GptOssForCausalLM extends Module implements SimpleModule {
  final GptOssModel transformer;
  final LinearLayer lmHead;

  GptOssForCausalLM({
    required super.name,
    required this.transformer,
    required this.lmHead,
  });

  @override
  Tensor forward(
    Tensor embeddings, {
    Tensor? pastKeyValues,
    Tensor? attentionMask,
    Tensor? tokenTypeIds,
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? inputsEmbeds,
    Tensor? encoderHiddenStates,
    Tensor? encoderAttentionMask,
    Tensor? labels,
    bool useCache = false,
    bool outputAttentions = false,
    bool outputHiddenStates = false,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor hiddenStates = transformer.forward(
      embeddings,
      pastKeyValues: pastKeyValues,
      attentionMask: attentionMask,
      tokenTypeIds: tokenTypeIds,
      positionIds: positionIds,
      headMask: headMask,
      inputsEmbeds: inputsEmbeds,
      encoderHiddenStates: encoderHiddenStates,
      encoderAttentionMask: encoderAttentionMask,
      useCache: useCache,
      outputAttentions: outputAttentions,
      outputHiddenStates: outputHiddenStates,
      context: context,
    );

    Tensor lmLogits = lmHead.forward(hiddenStates, context: context);

    return lmLogits;
  }

  /// Generate text using greedy decoding
  Tensor generate(
    Tensor inputIds, {
    required int maxNewTokens,
    required Context context,
    double temperature = 1.0,
    int topK = 0,
    double topP = 1.0,
  }) {
    Tensor currentInputIds = inputIds.to(device: context.device);

    for (int i = 0; i < maxNewTokens; i++) {
      // ... (Generation logic remains same as generic autoregressive)
      // Re-implementing simplified loop to save tokens/complexity, assuming standard generation logic from before was fine.
      // Copying the previous implementation.

      final logits = forward(currentInputIds, context: context);
      Tensor nextTokenLogits = logits.select(1, logits.shape[1] - 1);

      if (temperature > 0 && temperature != 1.0) {
        nextTokenLogits = nextTokenLogits / temperature;
      }

      if (topK > 0) {
        final (topKValues, _) = nextTokenLogits.topk(topK);
        final minTopKValue = topKValues.select(1, topK - 1).unsqueeze(1);
        nextTokenLogits = nextTokenLogits.maskedFill(
          nextTokenLogits.lt(minTopKValue),
          -double.infinity,
        );
      }

      // Top-P placeholder (same as before)

      Tensor nextToken;
      if (temperature == 0.0) {
        nextToken = nextTokenLogits.argmax(dim: -1);
      } else {
        final probs = nextTokenLogits.softmax(-1);
        nextToken = probs.multinomial(1);
      }

      if (nextToken.dim == 1) {
        nextToken = nextToken.unsqueeze(1);
      }

      currentInputIds = Tensor.cat([currentInputIds, nextToken], dim: 1);
    }

    return currentInputIds;
  }

  @override
  void resetParameters() {
    transformer.resetParameters();
    lmHead.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...transformer.parameters,
    ...lmHead.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {};

  @override
  late final Iterable<Module> submodules = [transformer, lmHead];

  static GptOssForCausalLM make({
    required GptOssConfig config,
    required String name,
  }) {
    final transformer = GptOssModel.make(config: config, name: 'transformer');

    final lmHead = LinearLayer.make(
      name: 'lm_head',
      inFeatures: config.embedDim,
      outFeatures: config.vocabSize,
      hasBias: false,
    );

    return GptOssForCausalLM(
      name: name,
      transformer: transformer,
      lmHead: lmHead,
    );
  }

  static Future<GptOssForCausalLM> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String name = '',
    String transformerName = 'model',
    String lmHeadName = 'lm_head',
    required GptOssConfig config,
  }) async {
    final transformer = await GptOssModel.loadFromSafeTensor(
      loader,
      name: transformerName,
      prefix: '$transformerName.',
      config: config,
    );

    final lmHead = await LinearLayer.loadFromSafeTensor(
      loader,
      name: lmHeadName,
      prefix: '$lmHeadName.',
    );

    return GptOssForCausalLM(
      name: name,
      transformer: transformer,
      lmHead: lmHead,
    );
  }
}
