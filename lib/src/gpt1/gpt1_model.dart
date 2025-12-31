import 'package:kamma/kamma.dart';
import 'gpt1_block.dart';
import 'gpt1_config.dart';

class OpenAIGPTModel extends Module {
  final EmbeddingLayer tokensEmbed;
  final EmbeddingLayer positionsEmbed;
  final Dropout drop;
  final List<OpenAIGPTBlock> h;

  OpenAIGPTModel({
    required super.name,
    required this.tokensEmbed,
    required this.positionsEmbed,
    required this.drop,
    required this.h,
  });

  Tensor forward(
    Tensor inputIds, {
    Tensor? attentionMask,
    Tensor?
    tokenTypeIds, // Not used in base model mostly, but kept for interface
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? inputsEmbeds,
    bool outputAttentions = false,
    bool outputHiddenStates = false,
    required Context context,
  }) {
    context.onloadModule(this);

    // Compute embeddings
    if (inputsEmbeds == null) {
      inputsEmbeds = tokensEmbed.forward(inputIds, context: context);
    }

    if (positionIds == null) {
      final seqLen = inputsEmbeds.shape[1]; // inputIds.shape[1]
      // Create position ids: [0, 1, ... seqLen-1]
      positionIds = Tensor.arange(
        0,
        seqLen,
        dataType: DataType.int64,
        device: inputsEmbeds.device,
      );
      // Expand to batch: [1, seqLen] -> [batch, seqLen]
      // But actually broadcasting usually works for addition.
      // However embedding layer expects [batch, seqLen] usually?
      // Or just [seqLen] if 1D.
      // inputsEmbeds is [batch, seqLen, hidden].
      // positionEmbeds needs to be broadcastable.
      // If we pass [seqLen] to embedding, we get [seqLen, hidden].
      // Then add [batch, seqLen, hidden] + [seqLen, hidden] -> Works.

      // Let's safe cast to batch if needed, but [seqLen] is fine.
      // positionIds = positionIds.unsqueeze(0);
    }

    Tensor positionEmbeds = positionsEmbed.forward(
      positionIds,
      context: context,
    );

    // Broadcast position embeddings if necessary
    if (positionEmbeds.shape.length == 2 && inputsEmbeds.shape.length == 3) {
      // positionEmbeds: [seqLen, hidden] or [1, seqLen, hidden]?
      // If positionIds was [seqLen], output is [seqLen, hidden].
      // We might need to unsqueeze to [1, seqLen, hidden] for broadcasting?
      // Tensor usually handles right-alignment broadcasting if broadcastable?
      // No, usually it matches last dims.
      // [batch, seqLen, hidden] + [seqLen, hidden] -> matches last 2 dims -> works.
    }

    Tensor hiddenStates = inputsEmbeds + positionEmbeds;

    // Convert tokenTypeIds to embeddings if we had them (not implemented in base OpenAIGPTModel as per standard HF, but DoubleHeads has it)

    hiddenStates = drop.forward(hiddenStates, context: context);

    for (final block in h) {
      hiddenStates = block.forward(
        hiddenStates,
        attentionMask: attentionMask,
        headMask: headMask,
        outputAttentions: outputAttentions,
        context: context,
      );
    }

    return hiddenStates;
  }

  @override
  void resetParameters() {
    tokensEmbed.resetParameters();
    positionsEmbed.resetParameters();
    for (final block in h) {
      block.resetParameters();
    }
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [
    tokensEmbed,
    positionsEmbed,
    drop,
    ...h,
  ];

  // Helpers
  int get embedDim => tokensEmbed.embeddingDim;
  int get vocabSize => tokensEmbed.numEmbeddings;
  int get nPositions => positionsEmbed.numEmbeddings;

  @override
  Map<String, dynamic> get meta => {
    "embedDim": embedDim,
    "vocabSize": vocabSize,
    "nPositions": nPositions,
  };

  static OpenAIGPTModel make(OpenAIGPTConfig config, {String name = 'model'}) {
    final tokensEmbed = EmbeddingLayer.make(
      name: 'tokens_embed',
      numEmbeddings: config.vocabSize,
      embedDim: config.nEmbd,
    );

    final positionsEmbed = EmbeddingLayer.make(
      name: 'positions_embed',
      numEmbeddings: config.nPositions,
      embedDim: config.nEmbd,
    );

    final drop = Dropout(config.embdPdrop);

    final h = <OpenAIGPTBlock>[];
    for (int i = 0; i < config.nLayer; i++) {
      h.add(
        OpenAIGPTBlock.make(
          name: 'h.$i',
          nCtx: config.nPositions, // Context size usually matches nPositions
          nEmbd: config.nEmbd,
          nHead: config.nHead,
          attnPdrop: config.attnPdrop,
          residPdrop: config.residPdrop,
          layerNormEpsilon: config.layerNormEpsilon,
          afn: config.afn,
        ),
      );
    }

    return OpenAIGPTModel(
      name: name,
      tokensEmbed: tokensEmbed,
      positionsEmbed: positionsEmbed,
      drop: drop,
      h: h,
    );
  }

  static Future<OpenAIGPTModel> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required OpenAIGPTConfig config,
    String prefix = '',
    String name = 'model',
  }) async {
    final tokensEmbed = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      name: 'tokens_embed',
      prefix: '${prefix}tokens_embed.',
    );

    final positionsEmbed = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      name: 'positions_embed',
      prefix: '${prefix}positions_embed.',
    );

    final drop = Dropout(config.embdPdrop);

    final h = <OpenAIGPTBlock>[];
    for (int i = 0; true; i++) {
      final blockPrefix = '${prefix}h.$i.';
      if (!loader.hasTensorWithPrefix(blockPrefix)) break;
      h.add(
        await OpenAIGPTBlock.loadFromSafeTensor(
          loader,
          prefix: blockPrefix,
          name: 'h.$i',
          nCtx: config.nPositions,
          nEmbd: config.nEmbd,
          nHead: config.nHead,
          attnPdrop: config.attnPdrop,
          residPdrop: config.residPdrop,
          layerNormEpsilon: config.layerNormEpsilon,
          afn: config.afn,
        ),
      );
    }

    return OpenAIGPTModel(
      name: name,
      tokensEmbed: tokensEmbed,
      positionsEmbed: positionsEmbed,
      drop: drop,
      h: h,
    );
  }
}
