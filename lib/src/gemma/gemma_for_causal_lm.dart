import 'package:kamma/kamma.dart';
import 'package:kamma/src/gemma/gemma_config.dart';
import 'package:kamma/src/gemma/gemma_model.dart';
import 'package:tensor/tensor.dart';

class GemmaForCausalLM extends Module {
  final GemmaModel model;

  /// The language modeling head.
  /// Note: In Gemma, weights are tied to embeddings.
  final LinearLayer lmHead;

  final GemmaConfig config;

  GemmaForCausalLM({
    required super.name,
    required this.model,
    required this.lmHead,
    required this.config,
  });

  static GemmaForCausalLM make({
    required GemmaConfig config,
    required Activation activation,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) {
    final model = GemmaModel.make(
      config: config,
      activation: activation,
      attentionMethod: attentionMethod,
    );

    // Tied weights
    // We create a LinearLayer but use the embedding weights.
    final lmHead = LinearLayer(
      name: 'lm_head',
      weight: model.embedTokens.weights,
      bias: null,
    );

    return GemmaForCausalLM(
      name: 'model', // usually just model or wrapped
      model: model,
      lmHead: lmHead,
      config: config,
    );
  }

  static Future<GemmaForCausalLM> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required GemmaConfig config,
    required Activation activation,
    GPT2AttentionMethodType attentionMethod = GPT2AttentionMethodType.sdap,
  }) async {
    final model = await GemmaModel.loadFromSafeTensor(
      loader,
      config: config,
      activation: activation,
      attentionMethod: attentionMethod,
    );

    // Reuse embedding weights for lm_head
    final lmHead = LinearLayer(
      name: 'lm_head',
      weight: model.embedTokens.weights,
      bias: null,
    );

    return GemmaForCausalLM(
      name: 'model',
      model: model,
      lmHead: lmHead,
      config: config,
    );
  }

  ({Tensor logits, List<Tensor>? allHiddenStates}) forward(
    Tensor inputIds, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    bool useCache = false,
    bool returnHiddenStates = false,
  }) {
    final output = model.forward(
      inputIds,
      context: context,
      attentionMask: attentionMask,
      positionIds: positionIds,
      useCache: useCache,
      returnHiddenStates: returnHiddenStates,
    );

    final hiddenStates = output.hiddenStates;

    final logits = lmHead.forward(hiddenStates, context: context);

    return (logits: logits, allHiddenStates: output.allHiddenStates);
  }

  void resetKeyValueCache() {
    model.resetKeyValueCache();
  }

  @override
  Iterable<Module> get submodules => [model, lmHead];

  @override
  Iterable<Tensor> get parameters => []; // Weights are in submodules

  @override
  void resetParameters() {
    model.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {};
}
