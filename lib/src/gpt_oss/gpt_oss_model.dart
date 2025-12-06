import 'package:tensor/tensor.dart';

class GptOssModel extends Module {
  final int embedDim;
  final int vocabSize;
  final int nPositions;
  final EmbeddingLayer wte;
  final Dropout drop;
  final List<GptOssDecoderLayer> h;
  final RMSNorm lnF;

  GptOssModel({
    required super.name,
    required this.embedDim,
    required this.vocabSize,
    required this.nPositions,
    required this.wte,
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

    // No positional embeddings added here; RoPE handles it in attention layers.
    inputsEmbeds ??= wte.forward(inputIds, context: context);

    Tensor hiddenStates = inputsEmbeds;
    hiddenStates = drop.forward(hiddenStates, context: context);

    for (final block in h) {
      hiddenStates = block.forward(
        hiddenStates,
        attentionMask: attentionMask,
        headMask: headMask,
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
    for (final block in h) {
      block.resetParameters();
    }
    lnF.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...wte.parameters,
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
  late final Iterable<Module> submodules = [wte, drop, ...h, lnF];

  static GptOssModel make({
    required GptOssConfig config,
    required String name,
  }) {
    final wte = EmbeddingLayer.make(config.vocabSize, config.nEmbd, name: 'wte')
      ..resetParameters(); // Ensure init

    final drop = Dropout(config.embdPdrop);

    final h = <GptOssDecoderLayer>[];
    for (int i = 0; i < config.nLayer; i++) {
      h.add(GptOssDecoderLayer.make(config: config, name: 'h.$i', layerIdx: i));
    }

    final lnF = RMSNorm.make(
      name: 'ln_f',
      normalizedShape: [config.nEmbd],
      eps: config.rmsNormEps,
    );

    return GptOssModel(
      name: name,
      embedDim: config.nEmbd,
      vocabSize: config.vocabSize,
      nPositions: config.nPositions,
      wte: wte,
      drop: drop,
      h: h,
      lnF: lnF,
    )..resetParameters();
  }

  Future<void> loadFromSafeTensor(SafeTensorLoader loader) async {
    await wte.loadFromSafeTensor(loader, prefix: 'wte.');
    // No wpe

    for (int i = 0; i < h.length; i++) {
      // blocks usually named h.0, h.1 or layers.0, layers.1
      // checking for h.0 first
      if (loader.hasTensor('h.$i.ln_1.weight') ||
          loader.hasTensor('h.$i.attn.c_attn.weight') ||
          loader.hasTensor('h.$i.ln_1.bias')) {
        await h[i].loadFromSafeTensor(loader, prefix: 'h.$i.');
      } else {
        // Fallback to 'layers.$i.' if appropriate, but keeping h.$i for now as primary
        await h[i].loadFromSafeTensor(loader, prefix: 'layers.$i.');
      }
    }

    // ln_f usually named ln_f or norm
    if (loader.hasTensor('ln_f.weight')) {
      await lnF.loadFromSafeTensor_(loader, prefix: 'ln_f.');
    } else if (loader.hasTensor('norm.weight')) {
      await lnF.loadFromSafeTensor_(loader, prefix: 'norm.');
    }
  }
}
