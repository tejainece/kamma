import 'package:kamma/kamma.dart';

class GPT2MLP extends Module implements SimpleModule {
  final LinearLayer cFc;
  final LinearLayer cProj;
  final Activation act;
  final Dropout dropout;

  GPT2MLP({
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

  int get innerDim => cFc.outFeatures;

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

  static GPT2MLP make({
    required String name,
    required int embedDim,
    required int? mlpInnerDim,
    required double residualDropoutProbability,
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

    // For now defaulting to GELU as per standard GPT-2
    final dropout = Dropout(residualDropoutProbability);

    return GPT2MLP(
      name: name,
      cFc: cFc,
      cProj: cProj,
      act: _makeActivation(),
      dropout: dropout,
    );
  }

  static Future<GPT2MLP> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required double residualDropoutProbability,
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
      cFc: cFc,
      cProj: cProj,
      act: _makeActivation(),
      dropout: dropout,
    );
  }

  static Activation _makeActivation() {
    // TODO: Handle different activation functions from config if needed
    return Activation.gelu;
  }
}
