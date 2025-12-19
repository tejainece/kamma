import 'package:kamma/kamma.dart';

class GPT2MLP extends Module implements SimpleModule {
  /// Projects the context-aware embedding output from [embedDim] to a higher [innerDim]
  /// to increase the expressive capacity.
  final LinearLayer upProjection;

  /// Projects back from [innerDim] to [embedDim] after applying the [activation] function.
  final LinearLayer downProjection;
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

  int get embedDim => upProjection.inFeatures;

  int get innerDim => upProjection.outFeatures;

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

    final cFc = LinearLayer.make(
      name: cFcName,
      inFeatures: embedDim,
      outFeatures: mlpInnerDim,
    );

    final cProj = LinearLayer.make(
      name: cProjName,
      inFeatures: mlpInnerDim,
      outFeatures: embedDim,
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
    final cFc = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '$prefix$cFcName.',
    );

    final cProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '$prefix$cProjName.',
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
