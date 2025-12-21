import 'package:kamma/kamma.dart';

class GPT2Model extends Module {
  final EmbeddingLayer wte;
  final EmbeddingLayer wpe;
  final Dropout drop;
  final List<GPT2Block> h;
  final LayerNorm lnF;

  GPT2Model({
    required super.name,
    required this.wte,
    required this.wpe,
    required this.drop,
    required this.h,
    required this.lnF,
  }) : assert(wte.embeddingDim == wpe.embeddingDim);

  Tensor forward(
    Tensor inputIds, {
    Tensor? pastKeyValues,
    Tensor? attentionMask,
    Tensor? tokenTypeIds,
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? inputsEmbeds,
    Tensor? encoderHiddenStates,
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
        0,
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
        outputAttentions: outputAttentions,
        context: context,
      );
    }

    hiddenStates = lnF.forward(hiddenStates, context: context);

    return hiddenStates;
  }

  void resetKeyValueCache(List<({Tensor? key, Tensor? value})>? keyValueCache) {
    for (int i = 0; i < h.length; i++) {
      h[i].resetKeyValueCache(
        key: keyValueCache?[i].key,
        value: keyValueCache?[i].value,
      );
    }
  }

  int get embedDim => wte.embeddingDim;

  int get vocabSize => wte.numEmbeddings;

  int get nPositions => wpe.numEmbeddings;

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
  late final Iterable<Tensor> parameters = [];

  @override
  Map<String, dynamic> get meta => {
    "embedDim": embedDim,
    "vocabSize": vocabSize,
    "nPositions": nPositions,
  };

  @override
  late final Iterable<Module> submodules = [wte, wpe, drop, ...h, lnF];

  static GPT2Model make({
    required String name,
    required double embedDropoutProbability,
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required int vocabSize,
    required int embedDim,
    required int numHeads,
    required int nPositions,
    required int nLayer,
    required double layerNormEpsilon,
    required bool isCrossAttention,
    required bool scaleAttnByInverseLayerIdx,
    required int maxPositionEmbeddings,
    required int? mlpInnerDim,
    required Activation activation,
    String wteName = 'wte',
    String wpeName = 'wpe',
    String lnFName = 'ln_f',
    String hName = 'h',
  }) {
    final wte = EmbeddingLayer.make(
      numEmbeddings: vocabSize,
      embedDim: embedDim,
      name: wteName,
    );

    final wpe = EmbeddingLayer.make(
      numEmbeddings: nPositions,
      embedDim: embedDim,
      name: wpeName,
    );

    final drop = Dropout(embedDropoutProbability);

    final h = <GPT2Block>[];
    for (int i = 0; i < nLayer; i++) {
      h.add(
        GPT2Block.make(
          name: '$hName.$i',
          layerIdx: i,
          attentionDropoutProbability: attentionDropoutProbability,
          residualDropoutProbability: residualDropoutProbability,
          embedDim: embedDim,
          numHeads: numHeads,
          layerNormEpsilon: layerNormEpsilon,
          isCrossAttention: isCrossAttention,
          scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
          maxPositionEmbeddings: maxPositionEmbeddings,
          mlpInnerDim: mlpInnerDim,
          activation: activation,
        ),
      );
    }

    final lnF = LayerNorm.make(
      name: lnFName,
      normalizedShape: [embedDim],
      eps: layerNormEpsilon,
    );

    return GPT2Model(
      name: name,
      wte: wte,
      wpe: wpe,
      drop: drop,
      h: h,
      lnF: lnF,
    );
  }

  static Future<GPT2Model> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String name,
    String prefix = '',
    String wteName = 'wte',
    String wpeName = 'wpe',
    String layerNormName = 'ln_f',
    required double embedDropoutProbability,
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required double layerNormEpsilon,
    required int numHeads,
    required bool scaleAttnByInverseLayerIdx,
    required int maxPositionEmbeddings,
    required Activation activation,
    required bool isCrossAttention,
  }) async {
    final wte = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      name: 'wte',
      prefix: '$prefix$wteName.',
    );
    final wpe = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      name: 'wpe',
      prefix: '$prefix$wpeName.',
    );
    final int embedDim = wte.embeddingDim;

    final blocks = <GPT2Block>[];
    for (int i = 0; true; i++) {
      final path = '${prefix}h.$i.';
      if (!loader.hasTensorWithPrefix(path)) break;
      final block = await GPT2Block.loadFromSafeTensor(
        loader,
        name: 'h.$i',
        prefix: path,
        layerNormEpsilon: layerNormEpsilon,
        attentionDropoutProbability: attentionDropoutProbability,
        residualDropoutProbability: residualDropoutProbability,
        numHeads: numHeads,
        isCrossAttention: isCrossAttention,
        layerIdx: i,
        scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
        maxPositionEmbeddings: maxPositionEmbeddings,
        activation: activation,
        embedDim: embedDim,
      );
      blocks.add(block);
    }
    final lnF = await LayerNorm.loadFromSafeTensor(
      loader,
      name: layerNormName,
      prefix: Module.combineDirs(prefix, layerNormName),
      normalizedShape: [embedDim],
    );
    return GPT2Model(
      name: name,
      wte: wte,
      wpe: wpe,
      drop: Dropout(embedDropoutProbability),
      h: blocks,
      lnF: lnF,
    );
  }
}
