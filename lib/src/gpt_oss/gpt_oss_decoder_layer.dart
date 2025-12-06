import 'package:kamma/kamma.dart';

/// TODO GradientCheckpointingLayer
class GptOssDecoderLayer extends Module implements SimpleModule {
  final RMSNorm ln1;
  final GptOssAttention attn;
  final RMSNorm ln2;
  final GptOssMoE moe;

  GptOssDecoderLayer({
    required super.name,
    required this.ln1,
    required this.attn,
    required this.ln2,
    required this.moe,
  });

  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? layerPast,
    Tensor? attentionMask,
    Tensor? headMask,
    Tensor? encoderHiddenStates,
    Tensor? encoderAttentionMask,
    bool useCache = false,
    bool outputAttentions = false,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor residual = hiddenStates;
    hiddenStates = ln1.forward(hiddenStates, context: context);

    Tensor attnOutput = attn.forward(
      hiddenStates,
      layerPast: layerPast,
      attentionMask: attentionMask,
      headMask: headMask,
      encoderHiddenStates: encoderHiddenStates,
      encoderAttentionMask: encoderAttentionMask,
      useCache: useCache,
      outputAttentions: outputAttentions,
      context: context,
    );

    hiddenStates = attnOutput + residual;

    residual = hiddenStates;
    hiddenStates = ln2.forward(hiddenStates, context: context);
    hiddenStates = moe.forward(hiddenStates, context: context);
    hiddenStates = hiddenStates + residual;

    return hiddenStates;
  }

  @override
  void resetParameters() {
    ln1.resetParameters();
    attn.resetParameters();
    ln2.resetParameters();
    moe.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...ln1.parameters,
    ...attn.parameters,
    ...ln2.parameters,
    ...moe.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {};

  @override
  late final Iterable<Module> submodules = [ln1, attn, ln2, moe];

  static GptOssDecoderLayer make({
    required String name,
    required int nEmbed,
    required double rmsNormEps,
    int layerIdx = 0,
  }) {
    final ln1 = RMSNorm.make(
      name: 'ln_1',
      normalizedShape: [nEmbed],
      eps: rmsNormEps,
    );

    final attention = GptOssAttention.make(name: 'attn', layerIdx: layerIdx);

    final ln2 = RMSNorm.make(
      name: 'ln_2',
      normalizedShape: [nEmbed],
      eps: rmsNormEps,
    );

    final moe = GptOssMoE.make(name: 'moe');

    return GptOssDecoderLayer(
      name: name,
      ln1: ln1,
      attn: attention,
      ln2: ln2,
      moe: moe,
    );
  }

  static Future<GptOssDecoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int nEmbed,
    required double rmsNormEps,
  }) async {
    // ln_1 is now RMSNorm, might be named 'input_layernorm' or 'ln_1' depending on mapping.
    // Assuming standard gpt-oss naming: input_layernorm
    RMSNorm ln1;
    String ln1Name;
    if (loader.hasTensor('${prefix}input_layernorm.weight')) {
      ln1Name = 'input_layernorm';
    } else {
      ln1Name = 'ln_1';
    }
    ln1 = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix$ln1Name.',
      name: ln1Name,
      normalizedShape: [nEmbed],
      eps: rmsNormEps,
    );

    GptOssAttention attention = await GptOssAttention.loadFromSafeTensor(
      loader,
      prefix: '${prefix}self_attn.',
    ); // attn usually self_attn

    // ln_2 -> post_attention_layernorm
    RMSNorm ln2;
    String ln2Name;
    if (loader.hasTensor('${prefix}post_attention_layernorm.weight')) {
      ln2Name = 'post_attention_layernorm';
    } else {
      ln2Name = 'ln_2';
    }
    ln2 = await RMSNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix$ln2Name.',
      name: ln2Name,
      normalizedShape: [nEmbed],
      eps: rmsNormEps,
    );
    if (loader.hasTensor('${prefix}post_attention_layernorm.weight')) {
      await ln2.loadFromSafeTensor_(
        loader,
        prefix: '${prefix}post_attention_layernorm.',
      );
    } else {
      await ln2.loadFromSafeTensor_(loader, prefix: '${prefix}ln_2.');
    }

    // mlp/moe
    GptOssMoE moe = await GptOssMoE.loadFromSafeTensor(
      loader,
      prefix: '${prefix}mlp.',
    ); // sometimes 'block_sparse_moe'

    return GptOssDecoderLayer(
      name: name,
      ln1: ln1,
      attn: attention,
      ln2: ln2,
      moe: moe,
    );
  }
}
