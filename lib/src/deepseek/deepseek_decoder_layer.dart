import 'package:tensor/tensor.dart';
import 'package:kamma/kamma.dart';
import 'deepseek_config.dart';
import 'deepseek_rms_norm.dart';
import 'deepseek_attention.dart';
import 'deepseek_moe.dart';
import 'deepseek_rotary_embedding.dart';

class DeepSeekDecoderLayer extends Module {
  final DeepSeekConfig config;
  final int layerIdx;

  // Modules
  late final DeepSeekV3MLA selfAttn;
  late final SimpleModule
  mlp; // Could be MoE or normal MLP in early layers? V3 uses MoE everywhere except first few?
  // V3 uses MoE from layer 4 onwards (technically layer index 3 if 0-indexed? or later?)
  // Actually usually first layer is dense, others MoE.
  // config.n_routed_experts > 0 implies MoE capable.
  // We can let the content decide or use a flag.
  // For V3 structure, let's assume all MoE for simplification or check config layer range.
  // "DeepSeek-V3 ... shared experts ... routed experts ..."
  // For the sake of this implementation, we will use DeepSeekV3MoE.
  // If config.n_routed_experts == 0, maybe revert to MLP.

  late final DeepSeekRMSNorm inputLayernorm;
  late final DeepSeekRMSNorm postAttentionLayernorm;

  DeepSeekDecoderLayer({
    required super.name,
    required this.config,
    required this.layerIdx,
    required DeepSeekRotaryEmbedding rotaryEmbedding,
  }) {
    selfAttn = DeepSeekV3MLA(
      name: '${name}.self_attn', // Standard naming
      config: config,
      layerIdx: layerIdx,
      rotaryEmbedding: rotaryEmbedding,
    );

    // MoE Check
    // If n_routed_experts > 0, use MoE.
    if (config.nRoutedExperts > 0) {
      mlp = DeepSeekV3MoE(
        name: '${name}.mlp',
        config: config,
        hiddenSize: config.hiddenSize,
        nSharExps: config.nSharedExperts,
        nRoutedExps: config.nRoutedExperts,
        topK: config.topK,
      );
    } else {
      // Fallback to dense MLP (shared expert logic effectively)
      // Re-use DeepSeekMLP from moe file
      mlp = DeepSeekMLP(
        name: '${name}.mlp',
        gateProj: LinearLayer.make(
          name: '${name}.mlp.gate_proj',
          inFeatures: config.hiddenSize,
          outFeatures: config.intermediateSize,
        ),
        upProj: LinearLayer.make(
          name: '${name}.mlp.up_proj',
          inFeatures: config.hiddenSize,
          outFeatures: config.intermediateSize,
        ),
        downProj: LinearLayer.make(
          name: '${name}.mlp.down_proj',
          inFeatures: config.intermediateSize,
          outFeatures: config.hiddenSize,
        ),
        hiddenAct: config.hiddenAct,
      );
    }

    inputLayernorm = DeepSeekRMSNorm(config);
    postAttentionLayernorm = DeepSeekRMSNorm(config);
  }

  @override
  // (Tensor hiddenStates, {Tensor? attentionMask, Tensor? positionIds, Tensor? pastKv, bool useCache})
  Tensor forward(
    Tensor hiddenStates, {
    required Context context,
    Tensor? attentionMask,
    Tensor? positionIds,
    Tensor? pastKv,
    bool useCache = false,
  }) {
    context.onloadModule(this);

    Tensor residual = hiddenStates;

    // 1. Input Norm
    Tensor h = inputLayernorm.forward(hiddenStates, context: context);

    // 2. Self Attention
    var attnOut = selfAttn.forward(
      h,
      context: context,
      attentionMask: attentionMask,
      positionIds: positionIds,
      pastKv: pastKv,
      useCache: useCache,
    );

    // Residual Add
    hiddenStates = residual + attnOut.output;

    // 3. Post Attention Norm & MLP
    residual = hiddenStates;
    h = postAttentionLayernorm.forward(hiddenStates, context: context);

    var mlpOut = mlp.forward(h, context: context);

    hiddenStates = residual + mlpOut;

    return hiddenStates;
  }

  @override
  Iterable<Module> get submodules => [
    inputLayernorm,
    selfAttn,
    postAttentionLayernorm,
    mlp,
  ];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  Iterable<Tensor> get nonTrainableParameters => [];

  @override
  void resetParameters() {
    for (var m in submodules) m.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {'layerIdx': layerIdx, 'config': config};
}
