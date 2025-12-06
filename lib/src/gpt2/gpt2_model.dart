import 'package:tensor/tensor.dart';

class GPT2Model extends Module {
  final int embedDim;
  final int vocabSize;
  final int nPositions;
  final EmbeddingLayer wte;
  final EmbeddingLayer wpe;
  final Dropout drop;
  final List<GPT2Block> h;
  final LayerNorm lnF;

  GPT2Model({
    required super.name,
    required this.embedDim,
    required this.vocabSize,
    required this.nPositions,
    required this.wte,
    required this.wpe,
    required this.drop,
    required this.h,
    required this.lnF,
  });

  @override
  Tensor forward(
    Tensor inputIds, {
    Tensor? pastKeyValues,
    Tensor? attentionMask,
    Tensor? tokenTypeIds,
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? inputsEmbeds,
    Tensor? encoderHiddenStates,
    Tensor? encoderAttentionMask,
    bool useCache = false,
    bool outputAttentions = false,
    bool outputHiddenStates = false,
    required Context context,
  }) {
    context.onloadModule(this);

    // Tensor inputShape = inputIds.shapeTensor;
    // TODO: Handle inputIds vs inputsEmbeds

    inputsEmbeds ??= wte.forward(inputIds, context: context);

    if (positionIds == null) {
      final inputShape = inputIds.shape;
      final seqLength = inputShape[1];

      // Create position ids [0, 1, ..., seqLength - 1]
      positionIds = Tensor.arange(
        seqLength,
        datatype: DataType.int64,
        device: inputIds.device,
      );

      // Expand to batch size: [batch_size, seqLength]
      positionIds = positionIds.unsqueeze(0).expand(inputShape);
    }

    Tensor positionEmbeds = wpe.forward(positionIds, context: context);

    if (tokenTypeIds != null) {
      // TODO: Add token type embeddings if wte has them or separate embedding layer
      // GPT2 usually doesn't use token type embeddings in the base model but some variants might
    }

    Tensor hiddenStates = inputsEmbeds + positionEmbeds;
    hiddenStates = drop.forward(hiddenStates, context: context);

    // TODO: Apply blocks
    for (final block in h) {
      hiddenStates = block.forward(
        hiddenStates,
        // layerPast: layerPast?[i],
        attentionMask: attentionMask,
        headMask: headMask, // TODO: split head mask per layer
        encoderHiddenStates: encoderHiddenStates,
        encoderAttentionMask: encoderAttentionMask,
        useCache: useCache,
        outputAttentions: outputAttentions,
        context: context,
      );
    }

    hiddenStates = lnF.forward(hiddenStates, context: context);

    return hiddenStates;
  }

  @override
  void resetParameters() {
    wte.resetParameters();
    wpe.resetParameters();
    // drop.resetParameters();
    for (final block in h) {
      block.resetParameters();
    }
    lnF.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...wte.parameters,
    ...wpe.parameters,
    ...h.expand((block) => block.parameters),
    ...lnF.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {
    "embedDim": embedDim,
    "vocabSize": vocabSize,
    "nPositions": nPositions,
  };

  @override
  late final Iterable<Module> submodules = [wte, wpe, drop, ...h, lnF];

  static GPT2Model make({required GPT2Config config, required String name}) {
    final wte = EmbeddingLayer.make(
      config.vocabSize,
      config.nEmbd,
      name: 'wte',
    );

    final wpe = EmbeddingLayer.make(
      config.nPositions,
      config.nEmbd,
      name: 'wpe',
    );

    final drop = Dropout(config.embdPdrop);

    final h = <GPT2Block>[];
    for (int i = 0; i < config.nLayer; i++) {
      h.add(GPT2Block.make(config: config, name: 'h.$i', layerIdx: i));
    }

    final lnF = LayerNorm.make(
      name: 'ln_f',
      normalizedShape: [config.nEmbd],
      eps: config.layerNormEpsilon,
    );

    return GPT2Model(
      name: name,
      embedDim: config.nEmbd,
      vocabSize: config.vocabSize,
      nPositions: config.nPositions,
      wte: wte,
      wpe: wpe,
      drop: drop,
      h: h,
      lnF: lnF,
    );
  }

  Future<void> loadFromSafeTensor(SafeTensorLoader loader) async {
    await wte.loadFromSafeTensor(loader, prefix: 'wte.');
    await wpe.loadFromSafeTensor(loader, prefix: 'wpe.');

    for (int i = 0; i < h.length; i++) {
      await h[i].loadFromSafeTensor(loader, prefix: 'h.$i.');
    }

    await lnF.loadFromSafeTensor_(loader, prefix: 'ln_f.');
  }
}
