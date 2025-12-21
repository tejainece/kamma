import 'package:kamma/kamma.dart';

class OpenAIGPTMLP extends Module implements SimpleModule {
  final LinearLayer cFc;
  final LinearLayer cProj;
  final Activation act;
  final Dropout dropout;

  OpenAIGPTMLP({
    required super.name,
    required this.cFc,
    required this.cProj,
    required this.act,
    required this.dropout,
  });

  @override
  Tensor forward(Tensor embeddings, {required Context context}) {
    context.onloadModule(this);

    embeddings = cFc.forward(embeddings, context: context);
    embeddings = act.forward(embeddings, context: context);
    embeddings = cProj.forward(embeddings, context: context);
    embeddings = dropout.forward(embeddings, context: context);

    return embeddings;
  }

  int get embedDim => cFc.numInFeatures;

  @override
  void resetParameters() {
    cFc.resetParameters();
    cProj.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  Map<String, dynamic> get meta => {"embedDim": embedDim};

  @override
  late final Iterable<Module> submodules = [cFc, cProj, dropout];

  static OpenAIGPTMLP make({
    required String name,
    required int nEmbd,
    required double residPdrop,
    String afn = "gelu",
  }) {
    final nState = 4 * nEmbd;

    final cFc = LinearLayer.make(
      name: 'c_fc',
      inFeatures: nEmbd,
      outFeatures: nState,
    );

    final cProj = LinearLayer.make(
      name: 'c_proj',
      inFeatures: nState,
      outFeatures: nEmbd,
    );

    final dropout = Dropout(residPdrop);

    return OpenAIGPTMLP(
      name: name,
      cFc: cFc,
      cProj: cProj,
      act: _makeActivation(afn),
      dropout: dropout,
    );
  }

  static Activation _makeActivation(String afn) {
    if (afn == "gelu") return Activation.gelu;
    if (afn == "relu") return Activation.relu;
    // Fallback or todo
    return Activation.gelu;
  }

  static Future<OpenAIGPTMLP> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required double residPdrop,
    String afn = "gelu",
  }) async {
    final cFc = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}c_fc.',
      name: 'c_fc',
    );

    final cProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}c_proj.',
      name: 'c_proj',
    );

    final dropout = Dropout(residPdrop);

    return OpenAIGPTMLP(
      name: name,
      cFc: cFc,
      cProj: cProj,
      act: _makeActivation(afn),
      dropout: dropout,
    );
  }
}
