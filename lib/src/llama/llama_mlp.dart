import 'package:tensor/tensor.dart';
import 'llama_config.dart';

class LlamaMLP extends Module implements SimpleModule {
  final LlamaConfig config;
  final int hiddenSize;
  final int intermediateSize;
  final LinearLayer gateProj;
  final LinearLayer upProj;
  final LinearLayer downProj;

  LlamaMLP(
    this.config, {
    required this.gateProj,
    required this.upProj,
    required this.downProj,
  }) : hiddenSize = config.hiddenSize,
       intermediateSize = config.intermediateSize,
       super(name: 'llama_mlp');

  static LlamaMLP make(LlamaConfig config) {
    final hiddenSize = config.hiddenSize;
    final intermediateSize = config.intermediateSize;

    return LlamaMLP(
      config,
      gateProj: LinearLayer.make(
        name: 'gate_proj',
        inFeatures: hiddenSize,
        outFeatures: intermediateSize,
        hasBias: config.mlpBias,
      ),
      upProj: LinearLayer.make(
        name: 'up_proj',
        inFeatures: hiddenSize,
        outFeatures: intermediateSize,
        hasBias: config.mlpBias,
      ),
      downProj: LinearLayer.make(
        name: 'down_proj',
        inFeatures: intermediateSize,
        outFeatures: hiddenSize,
        hasBias: config.mlpBias,
      ),
    );
  }

  static Future<LlamaMLP> loadFromSafeTensor(
    SafeTensorLoader loader,
    LlamaConfig config, {
    required String prefix,
  }) async {
    return LlamaMLP(
      config,
      gateProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}gate_proj.',
        name: 'gate_proj',
      ),
      upProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}up_proj.',
        name: 'up_proj',
      ),
      downProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}down_proj.',
        name: 'down_proj',
      ),
    );
  }

  @override
  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);

    final gate = gateProj.forward(x, context: context);
    final up = upProj.forward(x, context: context);

    Tensor activated;
    if (config.hiddenAct == 'silu') {
      activated = gate.silu();
    } else if (config.hiddenAct == 'relu') {
      activated = gate.relu();
    } else if (config.hiddenAct == 'gelu') {
      activated = gate.gelu(GeluApporimate.none);
    } else {
      // Fallback for swish or others
      if (config.hiddenAct == 'swish') {
        activated = gate.silu();
      } else {
        throw UnimplementedError(
          "Activation ${config.hiddenAct} not supported yet in LlamaMLP",
        );
      }
    }

    final intermediate = activated * up;
    return downProj.forward(intermediate, context: context);
  }

  @override
  Iterable<Module> get submodules => [gateProj, upProj, downProj];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  void resetParameters() {
    gateProj.resetParameters();
    upProj.resetParameters();
    downProj.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {
    'hiddenSize': hiddenSize,
    'intermediateSize': intermediateSize,
    'activation': config.hiddenAct,
  };
}
