import 'package:tensor/tensor.dart';

class LlamaMLP extends Module implements SimpleModule {
  final Activation activation;
  final LinearLayer gateProj;
  final LinearLayer upProj;
  final LinearLayer downProj;

  LlamaMLP({
    required super.name,
    required this.activation,
    required this.gateProj,
    required this.upProj,
    required this.downProj,
  });

  int get embedDim => gateProj.numInFeatures;

  int get intermediateSize => gateProj.numOutFeatures;

  @override
  Tensor forward(Tensor embeddings, {required Context context}) {
    context.onloadModule(this);

    Tensor input = embeddings;

    embeddings = gateProj.forward(embeddings, context: context);
    embeddings = activation.forward(embeddings, context: context);
    embeddings = embeddings * upProj.forward(input, context: context);
    embeddings = downProj.forward(embeddings, context: context);
    return embeddings;
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
    'embedDim': embedDim,
    'intermediateSize': intermediateSize,
    'activation': activation.name,
  };

  static LlamaMLP make({
    String name = 'mlp',
    required int embedDim,
    required int intermediateSize,
    required Activation activation,
    bool hasBias = false,
  }) {
    return LlamaMLP(
      name: name,
      activation: activation,
      gateProj: LinearLayer.make(
        name: 'gate_proj',
        inFeatures: embedDim,
        outFeatures: intermediateSize,
        hasBias: hasBias,
      ),
      upProj: LinearLayer.make(
        name: 'up_proj',
        inFeatures: embedDim,
        outFeatures: intermediateSize,
        hasBias: hasBias,
      ),
      downProj: LinearLayer.make(
        name: 'down_proj',
        inFeatures: intermediateSize,
        outFeatures: embedDim,
        hasBias: hasBias,
      ),
    );
  }

  static Future<LlamaMLP> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String name = 'mlp',
    required String prefix,
    required Activation activation,
  }) async {
    return LlamaMLP(
      name: name,
      activation: activation,
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
}
