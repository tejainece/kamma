import 'package:kamma/src/gpt_oss/gpt_oss.dart';
import 'package:tensor/tensor.dart';

class GptOssModel extends Module {
  final int nPositions;
  final EmbeddingLayer embedTokens;
  final Dropout drop;
  final List<GptOssDecoderLayer> heads;
  final RMSNorm norm;

  GptOssModel({
    required super.name,
    required this.nPositions,
    required this.embedTokens,
    required this.drop,
    required this.heads,
    required this.norm,
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

    // No positional embeddings added here; RoPE handles it in attention layers.
    inputsEmbeds ??= embedTokens.forward(inputIds, context: context);

    // Create default position IDs if not provided
    if (positionIds == null) {
      final seqLen = inputIds.shape[1];
      // simplified (assuming no past cache for defaults yet)
      positionIds = Tensor.arange(
        0,
        seqLen,
        datatype: DataType.int64,
        device: inputIds.device,
      ).unsqueeze(0).expand([inputIds.shape[0], seqLen]);
    }

    Tensor hiddenStates = inputsEmbeds;
    hiddenStates = drop.forward(hiddenStates, context: context);

    for (int i = 0; i < heads.length; i++) {
      final block = heads[i];
      // Handle cache splitting if pastKeyValues is implemented
      // List<Tensor>? layerPast = pastKeyValues?[i];

      hiddenStates = block.forward(
        hiddenStates,
        layerPast: null, // Placeholder until cache fixed
        attentionMask: attentionMask,
        positionIds: positionIds,
        headMask: headMask,
        encoderHiddenStates: encoderHiddenStates,
        encoderAttentionMask: encoderAttentionMask,
        useCache: useCache,
        outputAttentions: outputAttentions,
        context: context,
      );
    }

    hiddenStates = norm.forward(hiddenStates, context: context);

    return hiddenStates;
  }

  int get embedDim => embedTokens.embeddingDim;

  int get vocabSize => embedTokens.numEmbeddings;

  @override
  void resetParameters() {
    embedTokens.resetParameters();
    for (final block in heads) {
      block.resetParameters();
    }
    norm.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...embedTokens.parameters,
    ...heads.expand((block) => block.parameters),
    ...norm.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {
    "embedDim": embedDim,
    "vocabSize": vocabSize,
    "nPositions": nPositions,
  };

  @override
  late final Iterable<Module> submodules = [embedTokens, drop, ...heads, norm];

  static GptOssModel make({
    required GptOssConfig config,
    required String name,
    String embedTokensName = 'embed_tokens',
    String normName = 'norm',
    String headName = 'layers',
  }) {
    final embedTokens = EmbeddingLayer.make(
      numEmbeddings: config.vocabSize,
      embedDim: config.embedDim,
      name: embedTokensName,
    )..resetParameters(); // Ensure init

    final drop = Dropout(config.embdPdrop);

    final h = <GptOssDecoderLayer>[];
    for (int i = 0; i < config.nLayer; i++) {
      h.add(
        GptOssDecoderLayer.make(
          name: 'h.$i',
          layerIdx: i,
          embedDim: config.embedDim,
          numHeads: config.nHead,
          nInner: config.nInner,
          nPositions: config.nPositions,
          attentionDropoutP: config.attnPdrop,
          residDropoutP: config.residPdrop,
          numKeyValueHeads: config.numKeyValueHeads ?? config.nHead,
          ropeTheta: config.ropeTheta,
          isCrossAttention: false,
          numExperts: config.numExperts,
          numExpertsPerToken: config.numExpertsPerToken,
          rmsNormEps: config.rmsNormEps,
        ),
      );
    }

    final norm = RMSNorm.make(
      name: 'norm',
      normalizedShape: [config.embedDim],
      eps: config.rmsNormEps,
    );

    return GptOssModel(
      name: name,
      nPositions: config.nPositions,
      embedTokens: embedTokens,
      drop: drop,
      heads: h,
      norm: norm,
    )..resetParameters();
  }

  static Future<GptOssModel> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = 'model.',
    required String name,
    String embedTokensName = 'embed_tokens',
    String normName = 'norm',
    String headName = 'layers',
    required GptOssConfig config,
  }) async {
    final wte = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      prefix: '$prefix$embedTokensName.',
      name: embedTokensName,
    );

    final drop = Dropout(config.embdPdrop);

    final layers = <GptOssDecoderLayer>[];
    for (int i = 0; true; i++) {
      final path = '$prefix$headName.$i.';
      if (!loader.hasTensorWithPrefix(prefix)) break;
      layers.add(
        await GptOssDecoderLayer.loadFromSafeTensor(
          loader,
          name: '$headName.$i',
          prefix: path,
          attentionDropoutP: config.attnPdrop,
          residDropoutP: config.residPdrop,
          numKeyValueHeads: config.numKeyValueHeads ?? config.nHead,
          ropeTheta: config.ropeTheta,
          isCrossAttention: false,
          numExperts: config.numExperts,
          numExpertsPerToken: config.numExpertsPerToken,
          rmsNormEps: config.rmsNormEps,
          embedDim: config.embedDim,
          numHeads: config.nHead,
          nInner: config.nInner,
          nPositions: config.nPositions,
        ),
      );
    }

    final norm = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix$normName.',
      normalizedShape: [config.embedDim],
    );

    return GptOssModel(
      name: name,
      embedTokens: wte,
      drop: drop,
      heads: layers,
      norm: norm,
      nPositions: config.nPositions,
    );
  }
}
