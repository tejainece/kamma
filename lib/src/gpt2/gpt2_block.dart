import 'package:kamma/kamma.dart';

class GPT2Block extends Module implements SimpleModule {
  final LayerNorm ln1;
  final GPT2Attention attn;
  final LayerNorm ln2;
  final GPT2MLP mlp;

  GPT2Block({
    required super.name,
    required this.ln1,
    required this.attn,
    required this.ln2,
    required this.mlp,
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

    // TODO: Handle attnOutput being a tuple if useCache or outputAttentions is true
    // For now assuming it returns just the attention output tensor

    hiddenStates = attnOutput + residual;

    residual = hiddenStates;
    hiddenStates = ln2.forward(hiddenStates, context: context);
    hiddenStates = mlp.forward(hiddenStates, context: context);
    hiddenStates = hiddenStates + residual;

    return hiddenStates;
  }

  @override
  void resetParameters() {
    ln1.resetParameters();
    attn.resetParameters();
    ln2.resetParameters();
    mlp.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...ln1.parameters,
    ...attn.parameters,
    ...ln2.parameters,
    ...mlp.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {};

  @override
  late final Iterable<Module> submodules = [ln1, attn, ln2, mlp];

  static GPT2Block make({
    required GPT2Config config,
    required String name,
    int layerIdx = 0,
  }) {
    final ln1 = LayerNorm.make(
      name: 'ln_1',
      normalizedShape: [config.nEmbd],
      eps: config.layerNormEpsilon,
    );

    final attn = GPT2Attention.make(
      config: config,
      name: 'attn',
      layerIdx: layerIdx,
    );

    final ln2 = LayerNorm.make(
      name: 'ln_2',
      normalizedShape: [config.nEmbd],
      eps: config.layerNormEpsilon,
    );

    final mlp = GPT2MLP.make(config: config, name: 'mlp');

    return GPT2Block(name: name, ln1: ln1, attn: attn, ln2: ln2, mlp: mlp);
  }

  Future<void> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
  }) async {
    await ln1.loadFromSafeTensor_(loader, prefix: '${prefix}ln_1.');
    await attn.loadFromSafeTensor(loader, prefix: '${prefix}attn.');
    await ln2.loadFromSafeTensor_(loader, prefix: '${prefix}ln_2.');
    await mlp.loadFromSafeTensor(loader, prefix: '${prefix}mlp.');
  }
}
