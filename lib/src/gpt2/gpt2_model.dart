import 'package:kamma/kamma.dart';

import 'package:kamma/src/gpt2/causal_mask.dart';

class GPT2Model extends Module {
  final EmbeddingLayer wte;
  final EmbeddingLayer wpe;
  final Dropout drop;
  final List<GPT2Block> blocks;
  final LayerNorm lnF;

  GPT2Model({
    required super.name,
    required this.wte,
    required this.wpe,
    required this.drop,
    required this.blocks,
    required this.lnF,
  }) : assert(wte.embeddingDim == wpe.embeddingDim);

  /// [inputsEmbeds] is of shape (batchSize, seqLength, embedDim)
  Tensor forward(
    Tensor? inputIds, {
    Tensor? attentionMask,
    // TODO what is this used for
    Tensor? tokenTypeIds,
    Tensor? cachePositions,
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? inputsEmbeds,
    Tensor? encoderHiddenStates,
    List<Tensor>? allHiddenStates,
    List<Tensor>? outputAllSelfAttentions,
    List<Tensor>? outputAllCrossAttentions,
    required Context context,
  }) {
    context.onloadModule(this);

    int batchSize;
    if (inputIds != null && inputsEmbeds != null) {
      throw ArgumentError('inputIds and inputsEmbeds cannot both be provided');
    } else if (inputIds != null) {
      // TODO warn_if_padding_and_no_attention_mask
      final inputShape = inputIds.shape;
      inputIds = inputIds.view([-1, inputShape.last]);
      // TODO seqLength = inputShape.last;
      batchSize = inputIds.shape.first;
    } else if (inputsEmbeds != null) {
      batchSize = inputsEmbeds.shape.first;
      // TODO seqLength = inputsEmbeds.shape.last;
    } else {
      throw ArgumentError(
        'You have to specify either inputIds or inputsEmbeds',
      );
    }

    if (tokenTypeIds != null) {
      tokenTypeIds = tokenTypeIds.view([-1, tokenTypeIds.shape.last]);
    }

    inputsEmbeds ??= wte.forward(inputIds!, context: context);

    if (cachePositions == null) {
      int numProcessedTokens = blocks.first.attention.keyValueCache.seqLength;
      cachePositions = Tensor.arange(
        numProcessedTokens,
        numProcessedTokens + inputsEmbeds.shape[1],
        device: context.device,
      );
    }

    positionIds ??= cachePositions.unsqueeze(0); // TODO expand(inputShape);

    Tensor positionEmbeds = wpe.forward(positionIds, context: context);
    Tensor hiddenStates = inputsEmbeds + positionEmbeds;

    if (attentionMask != null) {
      attentionMask = attentionMask.view([batchSize, -1]);
    }

    final seqLength = inputIds?.shape[1] ?? inputsEmbeds!.shape[1];

    // Create causal mask
    Tensor? causalMask = createCausalMask(
      batchSize,
      seqLength,
      inputsEmbeds.dataType,
      context.device,
      method: blocks.first.attention.attentionMethod,
      pastKeyValuesLength: 0, // TODO support past key values
      attentionMask: attentionMask,
    );

    if (tokenTypeIds != null) {
      // TODO: Add token type embeddings if wte has them or separate embedding layer
      // GPT2 usually doesn't use token type embeddings in the base model but some variants might
    }

    hiddenStates = drop.forward(hiddenStates, context: context);

    // TODO compute head mask

    // TODO: Apply blocks
    for (final block in blocks) {
      // TODO implement parallelization
      allHiddenStates?.add(hiddenStates);
      hiddenStates = block.forward(
        hiddenStates,
        // TODO causal mask
        attentionMask: causalMask,
        headMask: headMask, // TODO: split head mask per layer
        encoderHiddenStates: encoderHiddenStates,
        outputSelfAttentions: outputAllSelfAttentions,
        outputCrossAttention: outputAllCrossAttentions,
        context: context,
      );
    }

    hiddenStates = lnF.forward(hiddenStates, context: context);
    // TODO we combined batchSize and choiceSize dimension of input. Lets not revert it back for output, so the caller gets expected output shape
    allHiddenStates?.add(hiddenStates);

    return hiddenStates;
  }

  void resetKeyValueCache(List<({Tensor? key, Tensor? value})>? keyValueCache) {
    for (int i = 0; i < blocks.length; i++) {
      blocks[i].resetKeyValueCache(
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
    for (final block in blocks) {
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
  late final Iterable<Module> submodules = [wte, wpe, drop, ...blocks, lnF];

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
      blocks: h,
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
      blocks: blocks,
      lnF: lnF,
    );
  }
}
