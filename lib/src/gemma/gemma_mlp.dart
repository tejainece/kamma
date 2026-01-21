import 'package:kamma/kamma.dart';
import 'package:tensor/tensor.dart';

class GemmaMLP extends Module {
  final LinearLayer gateProj;
  final LinearLayer upProj;
  final LinearLayer downProj;
  final Activation activation;

  GemmaMLP({
    required super.name,
    required this.gateProj,
    required this.upProj,
    required this.downProj,
    required this.activation,
  });

  static GemmaMLP make({
    required String name,
    required int hiddenSize,
    required int intermediateSize,
    required Activation activation,
    String gateProjName = 'gate_proj',
    String upProjName = 'up_proj',
    String downProjName = 'down_proj',
  }) {
    return GemmaMLP(
      name: name,
      gateProj: LinearLayer.make(
        name: gateProjName,
        inFeatures: hiddenSize,
        outFeatures: intermediateSize,
        hasBias: false,
      ),
      upProj: LinearLayer.make(
        name: upProjName,
        inFeatures: hiddenSize,
        outFeatures: intermediateSize,
        hasBias: false,
      ),
      downProj: LinearLayer.make(
        name: downProjName,
        inFeatures: intermediateSize,
        outFeatures: hiddenSize,
        hasBias: false,
      ),
      activation: activation,
    );
  }

  static Future<GemmaMLP> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required Activation activation,
    String gateProjName = 'gate_proj',
    String upProjName = 'up_proj',
    String downProjName = 'down_proj',
  }) async {
    return GemmaMLP(
      name: name,
      gateProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '$prefix$gateProjName.',
        name: gateProjName,
      ),
      upProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '$prefix$upProjName.',
        name: upProjName,
      ),
      downProj: await LinearLayer.loadFromSafeTensor(
        loader,
        prefix: '$prefix$downProjName.',
        name: downProjName,
      ),
      activation: activation,
    );
  }

  Tensor forward(Tensor x, {required Context context}) {
    context.onloadModule(this);

    // down_proj(act(gate_proj(x)) * up_proj(x))
    final gate = gateProj.forward(x, context: context);
    final up = upProj.forward(x, context: context);

    // Gemma uses GeGLU:
    // x = act(gate) * up
    // But act depends on config (usually gelu or gelu_pytorch_tanh)
    // We apply activation to gate.

    // Assuming activation handles Tensor input and returns Tensor
    final activatedGate = activation.forward(gate, context: context);

    final intermediate = activatedGate * up;
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
  Map<String, dynamic> get meta => {};
}
