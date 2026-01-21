import 'package:tensor/tensor.dart';
import 'package:kamma/kamma.dart';

class GemmaRMSNorm extends Module {
  final double eps;
  Tensor weight;

  GemmaRMSNorm(this.weight, {this.eps = 1e-6}) : super(name: 'rms_norm');

  static GemmaRMSNorm make({
    required String name,
    required List<int> normalizedShape,
    double eps = 1e-5,
  }) {
    // Gemma initializes weights to zeros, but since we use (1 + weight),
    // and standard RMSNorm uses weight initialized to 1,
    // we should be careful.
    // Transformers implementation:
    // self.weight = nn.Parameter(torch.zeros(dim))
    // forward: ... * (1 + self.weight)

    // In our case, Tensor doesn't support 'zeros' init easily for Module loading maybe?
    // Let's assume we initialize to zeros.
    return GemmaRMSNorm(Tensor.zeros(normalizedShape), eps: eps)..name = name;
  }

  static Future<GemmaRMSNorm> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required List<int> normalizedShape,
    required double eps,
  }) async {
    final weight = await loader.loadByName('${prefix}weight');
    return GemmaRMSNorm(weight, eps: eps)..name = name;
  }

  Tensor forward(Tensor input, {required Context context}) {
    context.onloadModule(this);

    // Calculate RMS
    // input is float32 usually for exactness, but lets follow input dtype or cast if needed.
    // LlamaRMSNorm casts to float32.
    final inputFloat = input.to(dataType: DataType.float32);
    final variance = inputFloat.pow(2).mean(dim: [-1], keepDim: true);
    final hiddenStates = inputFloat * (variance + eps).rsqrt();

    // (1 + weight) * hiddenStates
    // We cast back to input type? Or weight type?
    // Usually weight and input are same type.

    return hiddenStates.to(dataType: input.dataType) * (weight + 1.0);
  }

  @override
  Iterable<Tensor> get parameters => [weight];

  @override
  void resetParameters() {
    // Initialized to zeros
    weight = Tensor.zeros(
      weight.shape,
      dataType: weight.dataType,
      device: weight.device,
    );
    // But since we can't easily mutate tensor data in place without assignments, we might need a better way if this was mutable.
    // For inference it's fine.
  }

  @override
  Iterable<Module> get submodules => [];

  @override
  Map<String, dynamic> get meta => {'eps': eps};
}
