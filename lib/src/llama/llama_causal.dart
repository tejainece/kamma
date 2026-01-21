import 'package:kamma/kamma.dart';
import 'package:kamma/src/common/rope/rope.dart';

class LlamaForCausalLM extends Module {
  final LlamaModel model;
  final LinearLayer lmHead;
  final int vocabSize;

  LlamaForCausalLM({
    required this.model,
    required this.lmHead,
    required this.vocabSize,
  }) : super(name: 'llama');

  ({Tensor logits, List<Tensor>? hiddenStates}) forward(
    Tensor embeddings, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    bool returnHiddenStates = false,
  }) {
    context.onloadModule(this);

    final result = model.forward(
      embeddings,
      context: context,
      attentionMask: attentionMask,
      positionIds: positionIds,
      returnHiddenStates: returnHiddenStates,
    );

    final logits = lmHead.forward(result.hiddenStates, context: context);
    return (logits: logits, hiddenStates: result.allHiddenStates);
  }

  @override
  Iterable<Module> get submodules => [model, lmHead];

  @override
  Iterable<Tensor> get parameters => []; // Params in submodules

  @override
  void resetParameters() {
    model.resetParameters();
    lmHead.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {};

  static LlamaForCausalLM make(
    LlamaConfig config, {
    // TODO change this to sdpa
    GPT2AttentionMethodType attentionMethod = .eager,
  }) {
    final activation = config.hiddenAct == 'silu'
        ? Activation.silu
        : Activation.gelu;

    return LlamaForCausalLM(
      vocabSize: config.vocabSize,
      model: LlamaModel.make(
        name: 'model',
        numLayers: config.numHiddenLayers,
        embedDim: config.hiddenSize,
        numHeads: config.numAttentionHeads,
        numKeyValueHeads: config.numKeyValueHeads,
        intermediateSize: config.intermediateSize,
        activation: activation,
        rmsNormEps: config.rmsNormEps,
        ropeTheta: config.ropeTheta,
        maxPositionEmbeddings: config.maxPositionEmbeddings,
        ropeArgs: config.ropeScaling != null
            ? RopeArgs.parse(config.ropeScaling!)
            : DefaultRopeArgs.instance,
        vocabSize: config.vocabSize,
        hasAttentionBias: config.attentionBias,
        attentionDropoutProb: config.attentionDropout,
        padTokenId: config.padTokenId,
        isCausal: true,
        attentionMethod: attentionMethod,
      ),
      lmHead: LinearLayer.make(
        name: 'lm_head',
        inFeatures: config.hiddenSize,
        outFeatures: config.vocabSize,
        hasBias: false,
      ),
    );
  }

  static Future<LlamaForCausalLM> loadFromSafeTensor(
    SafeTensorLoader loader,
    LlamaConfig config, {
    // TODO change this to sdpa
    GPT2AttentionMethodType attentionMethod = .eager,
  }) async {
    final activation = config.hiddenAct == 'silu'
        ? Activation.silu
        : Activation.gelu;
    final model = await LlamaModel.loadFromSafeTensor(
      loader,
      numLayers: config.numHiddenLayers,
      embedDim: config.hiddenSize,
      numHeads: config.numAttentionHeads,
      numKeyValueHeads: config.numKeyValueHeads,
      intermediateSize: config.intermediateSize,
      activation: activation,
      rmsNormEps: config.rmsNormEps,
      ropeTheta: config.ropeTheta,
      maxPositionEmbeddings: config.maxPositionEmbeddings,
      ropeArgs: config.ropeScaling != null
          ? RopeArgs.parse(config.ropeScaling!)
          : DefaultRopeArgs.instance,
      vocabSize: config.vocabSize,
      attentionDropoutProb: config.attentionDropout,
      padTokenId: config.padTokenId,
      isCausal: true,
      attentionMethod: attentionMethod,
    );
    LinearLayer lmHead;
    if (config.tieWordEmbeddings) {
      // Reuse embedding weights
      lmHead = LinearLayer(
        name: 'lm_head',
        weight: model.tokens.weights,
        bias: null,
      );
    } else {
      lmHead = await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: 'lm_head.',
        name: 'lm_head',
      );
    }
    return LlamaForCausalLM(
      model: model,
      lmHead: lmHead,
      vocabSize: config.vocabSize,
    );
  }

  void resetKeyValueCache() {
    model.resetKeyValueCache();
  }
}
