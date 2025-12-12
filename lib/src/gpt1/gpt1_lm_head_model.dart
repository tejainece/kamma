import 'package:kamma/kamma.dart';
import 'gpt1_config.dart';
import 'gpt1_model.dart';

class OpenAIGPTLMHeadModel extends Module {
  final OpenAIGPTModel transformer;
  final LinearLayer lmHead;

  OpenAIGPTLMHeadModel({
    required super.name,
    required this.transformer,
    required this.lmHead,
  });

  Tensor forward(
    Tensor inputIds, {
    Tensor? attentionMask,
    Tensor? tokenTypeIds,
    Tensor? positionIds,
    Tensor? headMask,
    Tensor? inputsEmbeds,
    bool outputAttentions = false,
    bool outputHiddenStates = false,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor hiddenStates = transformer.forward(
      inputIds,
      attentionMask: attentionMask,
      tokenTypeIds: tokenTypeIds,
      positionIds: positionIds,
      headMask: headMask,
      inputsEmbeds: inputsEmbeds,
      outputAttentions: outputAttentions,
      outputHiddenStates: outputHiddenStates,
      context: context,
    );

    Tensor lmLogits = lmHead.forward(hiddenStates, context: context);

    return lmLogits;
  }

  @override
  void resetParameters() {
    transformer.resetParameters();
    lmHead.resetParameters();
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [transformer, lmHead];

  @override
  Map<String, dynamic> get meta => transformer.meta;

  static OpenAIGPTLMHeadModel make(
    OpenAIGPTConfig config, {
    String name = 'model',
  }) {
    final transformer = OpenAIGPTModel.make(config, name: 'transformer');

    // LM Head
    // Weights are tied to embedding
    // We create LinearLayer but we should ideally share weights.
    // In 'make', we initialize random weights. Sharing requires extracting embedding weight.

    // For strictly following transformers "structure", we create the module.
    // Tying is usually done after instantiation or explicitly handling it.
    // Here we can share the weight tensor if we access it.

    // We construct LinearLayer manually to share weights
    final lmHead = LinearLayer(
      name: 'lm_head',
      weight: transformer.tokensEmbed.weights,
      bias: null,
    );

    return OpenAIGPTLMHeadModel(
      name: name,
      transformer: transformer,
      lmHead: lmHead,
    );
  }

  static Future<OpenAIGPTLMHeadModel> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required OpenAIGPTConfig config,
    String prefix = '',
    String name = 'model',
  }) async {
    final transformer = await OpenAIGPTModel.loadFromSafeTensor(
      loader,
      config: config,
      prefix: '${prefix}transformer.',
      name: 'transformer',
    );

    // When loading, if lm_head weights are not in file (because tied), we use embedding weights.
    // Standard HF saving saves tied weights too usually? Or maybe not.
    // If we assume strict loading:
    // Try to load 'lm_head' weights. If not found, use embedding.

    LinearLayer lmHead;
    if (loader.hasTensorWithPrefix('${prefix}lm_head')) {
      lmHead = await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}lm_head.',
        name: 'lm_head',
      );
    } else {
      lmHead = LinearLayer(
        name: 'lm_head',
        weight: transformer.tokensEmbed.weights,
        bias: null,
      );
    }

    return OpenAIGPTLMHeadModel(
      name: name,
      transformer: transformer,
      lmHead: lmHead,
    );
  }
}
