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
  Tensor forward(Tensor hiddenStates, {required Context context}) {
    context.onloadModule(this);

    hiddenStates = cFc.forward(hiddenStates, context: context);
    hiddenStates = act.forward(hiddenStates, context: context);
    hiddenStates = cProj.forward(hiddenStates, context: context);
    hiddenStates = dropout.forward(hiddenStates, context: context);

    return hiddenStates;
  }

  int get embedDim => cFc.inFeatures;

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
