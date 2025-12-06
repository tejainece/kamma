import 'package:kamma/kamma.dart';

/// Feed-forward network (MLP) for S3-DiT
///
/// Two-layer structure with intermediate dimension 2.5x hidden size,
/// regularized by RMSNorm.
class S3DiTMLP extends Module {
  final int hiddenSize;
  final int intermediateSize;
  final String activation;

  final LinearLayer fc1;
  final LinearLayer fc2;
  final Activation activationFn;
  final Dropout dropout;

  S3DiTMLP({
    required super.name,
    required this.hiddenSize,
    required this.intermediateSize,
    required this.activation,
    required this.fc1,
    required this.fc2,
    required this.activationFn,
    required this.dropout,
  });

  @override
  Tensor forward(Tensor hiddenStates, {required Context context}) {
    context.onloadModule(this);

    // First linear layer
    Tensor x = fc1.forward(hiddenStates, context: context);

    // Activation
    x = activationFn.forward(x, context: context);

    // Dropout
    x = dropout.forward(x, context: context);

    // Second linear layer
    x = fc2.forward(x, context: context);

    // Dropout
    x = dropout.forward(x, context: context);

    return x;
  }

  @override
  void resetParameters() {
    fc1.resetParameters();
    fc2.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...fc1.parameters,
    ...fc2.parameters,
  ];

  @override
  late final Iterable<Module> submodules = [fc1, fc2, dropout];

  @override
  Map<String, dynamic> get meta => {
    'hiddenSize': hiddenSize,
    'intermediateSize': intermediateSize,
    'activation': activation,
  };

  static S3DiTMLP make({required S3DiTConfig config, required String name}) {
    final fc1 = LinearLayer.make(
      name: 'fc1',
      inFeatures: config.hiddenSize,
      outFeatures: config.intermediateSize,
    );

    final fc2 = LinearLayer.make(
      name: 'fc2',
      inFeatures: config.intermediateSize,
      outFeatures: config.hiddenSize,
    );

    Activation activationFn;
    switch (config.hiddenActivation.toLowerCase()) {
      case 'gelu':
        activationFn = GeluActivation();
        break;
      case 'silu':
      case 'swish':
        activationFn = SiLU();
        break;
      default:
        throw ArgumentError(
          'Unsupported activation: ${config.hiddenActivation}',
        );
    }

    final dropout = Dropout(config.hiddenDropout);

    return S3DiTMLP(
      name: name,
      hiddenSize: config.hiddenSize,
      intermediateSize: config.intermediateSize,
      activation: config.hiddenActivation,
      fc1: fc1,
      fc2: fc2,
      activationFn: activationFn,
      dropout: dropout,
    );
  }
}
