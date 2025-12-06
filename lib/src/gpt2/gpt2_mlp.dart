import 'package:kamma/kamma.dart';

class GPT2MLP extends Module {
  final int embedDim;
  final LinearLayer cFc;
  final LinearLayer cProj;
  final Activation act;
  final Dropout dropout;

  GPT2MLP({
    required super.name,
    required this.embedDim,
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

  @override
  void resetParameters() {
    cFc.resetParameters();
    cProj.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...cFc.parameters,
    ...cProj.parameters,
  ];

  @override
  Map<String, dynamic> get meta => {"embedDim": embedDim};

  @override
  late final Iterable<Module> submodules = [cFc, cProj, dropout];

  static GPT2MLP make({required GPT2Config config, required String name}) {
    final embedDim = config.nEmbd;
    final innerDim = config.nInner > 0 ? config.nInner : 4 * embedDim;

    final cFc = LinearLayer.make(
      name: 'c_fc',
      inFeatures: embedDim,
      outFeatures: innerDim,
    );

    final cProj = LinearLayer.make(
      name: 'c_proj',
      inFeatures: innerDim,
      outFeatures: embedDim,
    );

    // TODO: Handle different activation functions from config if needed
    // For now defaulting to GELU as per standard GPT-2
    const act = Activation.gelu;
    final dropout = Dropout(config.residPdrop);

    return GPT2MLP(
      name: name,
      embedDim: embedDim,
      cFc: cFc,
      cProj: cProj,
      act: act,
      dropout: dropout,
    );
  }

  Future<void> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
  }) async {
    // Load c_fc
    if (loader.hasTensor('${prefix}c_fc.weight')) {
      final weight = await loader.loadByName('${prefix}c_fc.weight');
      cFc.weight.copy_(weight.transpose(0, 1));
    }
    if (loader.hasTensor('${prefix}c_fc.bias')) {
      final bias = await loader.loadByName('${prefix}c_fc.bias');
      cFc.bias!.copy_(bias);
    }

    // Load c_proj
    if (loader.hasTensor('${prefix}c_proj.weight')) {
      final weight = await loader.loadByName('${prefix}c_proj.weight');
      cProj.weight.copy_(weight.transpose(0, 1));
    }
    if (loader.hasTensor('${prefix}c_proj.bias')) {
      final bias = await loader.loadByName('${prefix}c_proj.bias');
      cProj.bias!.copy_(bias);
    }
  }
}
