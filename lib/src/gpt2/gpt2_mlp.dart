import 'package:tensor/tensor.dart';

class GPT2MLP extends Module implements SimpleModule {
  /// Projects the context-aware embedding output from [embedDim] to a higher [innerDim]
  /// to increase the expressive capacity.
  final LinearTransposed upProjection;

  /// Projects back from [innerDim] to [embedDim] after applying the [activation] function.
  final LinearTransposed downProjection;
  final Activation activation;
  final Dropout dropout;

  GPT2MLP({
    required super.name,
    required this.upProjection,
    required this.downProjection,
    required this.activation,
    required this.dropout,
  });

  @override
  Tensor forward(Tensor embeddings, {required Context context}) {
    context.onloadModule(this);

    embeddings = upProjection.forward(embeddings, context: context);
    embeddings = activation.forward(embeddings, context: context);
    embeddings = downProjection.forward(embeddings, context: context);
    embeddings = dropout.forward(embeddings, context: context);

    return embeddings;
  }

  int get embedDim => upProjection.numInFeatures;

  int get innerDim => upProjection.numOutFeatures;

  @override
  void resetParameters() {
    upProjection.resetParameters();
    downProjection.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  Map<String, dynamic> get meta => {"embedDim": embedDim};

  @override
  late final Iterable<Module> submodules = [
    upProjection,
    downProjection,
    dropout,
  ];

  static GPT2MLP make({
    required String name,
    required int embedDim,
    required int? mlpInnerDim,
    required double residualDropoutProbability,
    required Activation activation,
    String cFcName = 'c_fc',
    String cProjName = 'c_proj',
  }) {
    mlpInnerDim ??= 4 * embedDim;

    final cFc = LinearTransposed.make(
      name: cFcName,
      numInFeatures: embedDim,
      numOutFeatures: mlpInnerDim,
    );

    final cProj = LinearTransposed.make(
      name: cProjName,
      numInFeatures: mlpInnerDim,
      numOutFeatures: embedDim,
    );

    final dropout = Dropout(residualDropoutProbability);

    return GPT2MLP(
      name: name,
      upProjection: cFc,
      downProjection: cProj,
      activation: activation,
      dropout: dropout,
    );
  }

  static Future<GPT2MLP> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required double residualDropoutProbability,
    required Activation activation,
    String cFcName = 'c_fc',
    String cProjName = 'c_proj',
  }) async {
    final cFc = await LinearTransposed.loadFromSafeTensor(
      loader,
      prefix: '$prefix$cFcName.',
      name: cFcName,
    );

    final cProj = await LinearTransposed.loadFromSafeTensor(
      loader,
      prefix: '$prefix$cProjName.',
      name: cProjName,
    );

    final dropout = Dropout(residualDropoutProbability);

    return GPT2MLP(
      name: name,
      upProjection: cFc,
      downProjection: cProj,
      activation: activation,
      dropout: dropout,
    );
  }
}
