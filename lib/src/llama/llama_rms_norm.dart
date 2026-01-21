import 'package:tensor/tensor.dart';

class LlamaRMSNorm extends Module implements SimpleModule {
  final double eps;
  final Tensor weight; // Llama RMSNorm only has weight (gamma), no bias (beta)

  LlamaRMSNorm({required super.name, required this.eps, required this.weight});

  @override
  Tensor forward(Tensor input, {required Context context}) {
    context.onloadModule(this);

    // Llama RMSNorm:
    // 1. Cast to float32 (if needed, usually safe to do ops in float32 for precision)
    // 2. var = mean(x^2)
    // 3. rsqrt(var + eps)
    // 4. x * rsqrt * weight

    // Note: Tensor operations in Dart/Tensor usually propagate types.
    // For now assuming input is float32 or ops handle it.

    // Cast to float32 for precision
    final inputFloat = input.to(dataType: DataType.float32);

    // x^2
    final pows = inputFloat.pow(2.0);
    // mean(-1)
    final mean = pows.mean(dim: [-1], keepDim: true);
    // + eps
    final meanEps = mean + eps;
    // rsqrt
    final rsqrt = meanEps.rsqrt(); // or .pow(-0.5)

    // norm = x * rsqrt
    final norm = inputFloat * rsqrt;

    // output = norm * weight
    // Cast back to input type if needed? PyTorch does.
    // "return self.weight * hidden_states.to(input_dtype)"
    // Assuming weight is also compatible or will broadcast.
    // Let's cast back to input.dataType
    return (norm * weight).to(dataType: input.dataType);
  }

  @override
  Iterable<Module> get submodules => [];

  @override
  Iterable<Tensor> get parameters => [weight];

  @override
  void resetParameters() {
    // Usually initialized to 1.0
    // weight.fill(1.0); // If mutable
  }

  @override
  Map<String, dynamic> get meta => {'eps': eps};

  static Future<LlamaRMSNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int normalizedShape, // Dimension
    double eps = 1e-6,
  }) async {
    final weight = await loader.loadByName(
      '${prefix}weight',
      device: Device.cpu, // Should match model device
    );
    return LlamaRMSNorm(name: name, eps: eps, weight: weight);
  }

  // Factory for creation (random init)
  static LlamaRMSNorm make({
    required String name,
    required int dim,
    double eps = 1e-6,
  }) {
    return LlamaRMSNorm(name: name, eps: eps, weight: Tensor.ones([dim]));
  }
}
