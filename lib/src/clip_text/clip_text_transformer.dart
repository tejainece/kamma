import 'package:kamma/kamma.dart';

abstract class TextEncoder {}

class ClipTextTransformer extends Module implements TextEncoder {
  final ClipTextEmbeddings embeddings;
  final ClipEncoder encoder;
  final LayerNorm norm;

  ClipTextTransformer({
    required super.name,
    required this.embeddings,
    required this.encoder,
    required this.norm,
  });

  Tensor forward(
    Tensor inputIds, {
    Tensor? attentionMask,
    Tensor? positionIds,
    required Context context,
  }) {
    inputIds = inputIds.view([-1, inputIds.shape.last]);
    final hiddenStates = embeddings.forward(
      inputIds,
      positionIds: positionIds,
      context: context,
    );
    final attentionMaskMade = createCausalMask(
      inputEmbeds: hiddenStates,
      attentionMask: attentionMask,
      cachePosition: Tensor.arange(
        0,
        hiddenStates.shape[1],
        device: hiddenStates.device,
      ),
    );
    Tensor lastHiddenState = encoder.forward(
      hiddenStates,
      attentionMask: attentionMaskMade,
      context: context,
    );
    lastHiddenState = norm.forward(lastHiddenState, context: context);

    /* TODO pooled layer
    Tensor pooledOutput = lastHiddenState.index([
      Tensor.arange(lastHiddenState.shape[0]),
      
    ]);*/

    return lastHiddenState;
  }

  @override
  void resetParameters() {
    // TODO
    throw UnimplementedError();
  }

  @override
  late final Map<String, dynamic> meta = {
    'embeddings': embeddings.meta,
    'norm': norm.meta,
  };

  @override
  final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [embeddings, encoder, norm];

  static Future<ClipTextTransformer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required ClipTextConfig config,
    String prefix = 'text_model.',
    required String name,
  }) async {
    final embeddings = await ClipTextEmbeddings.loadFromSafeTensor(
      loader,
      prefix: '${prefix}embeddings.',
      name: 'embeddings',
    );
    final encoder = await ClipEncoder.loadFromSafeTensor(
      loader,
      prefix: '${prefix}encoder.',
      name: 'encoder',
      config: config,
    );
    final norm = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}final_layer_norm.',
      name: 'final_layer_norm',
      eps: config.layerNormEps,
      normalizedShape: [config.embedDim],
    );
    return ClipTextTransformer(
      name: name,
      embeddings: embeddings,
      encoder: encoder,
      norm: norm,
    );
  }
}

dynamic _processMaskArguments() {}

dynamic createCausalMask({
  required Tensor inputEmbeds,
  Tensor? attentionMask,
  required Tensor cachePosition,
  Tensor? positionIds,
}) {
  // TODO
}

class ClipEncoder extends Module implements SimpleModule {
  final List<ClipEncoderLayer> layers;

  ClipEncoder({required super.name, required this.layers});

  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? attentionMask,
    required Context context,
  }) {
    for (final layer in layers) {
      hiddenStates = layer.forward(
        hiddenStates,
        attentionMask: attentionMask,
        context: context,
      );
    }
    return hiddenStates;
  }

  @override
  final Map<String, dynamic> meta = const {};

  @override
  void resetParameters() {
    for (final layer in layers) {
      layer.resetParameters();
    }
  }

  @override
  final Iterable<Tensor> parameters = [];

  @override
  final Iterable<Module> submodules = [];

  static Future<ClipEncoder> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = 'text_model.encoder.',
    required ClipTextConfig config,
    required String name,
  }) async {
    final layers = <ClipEncoderLayer>[];
    int layerId = 0;
    while (true) {
      if (!loader.hasTensor('${prefix}layer.$layerId')) {
        break;
      }
      final layer = await ClipEncoderLayer.loadFromSafeTensor(
        loader,
        prefix: '${prefix}layer.$layerId.',
        name: 'layer.$layerId',
        config: config,
      );
      layers.add(layer);
      layerId++;
    }
    return ClipEncoder(name: name, layers: layers);
  }
}

class ClipEncoderLayer extends Module implements SimpleModule {
  final ClipAttention selfAttention;
  final LayerNorm layerNorm1;
  final ClipMlp mlp;
  final LayerNorm layerNorm2;

  ClipEncoderLayer({
    required super.name,
    required this.selfAttention,
    required this.layerNorm1,
    required this.mlp,
    required this.layerNorm2,
  });

  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? attentionMask,
    required Context context,
  }) {
    Tensor residual = hiddenStates;
    hiddenStates = layerNorm1.forward(hiddenStates, context: context);
    (hiddenStates, _) = selfAttention.forward(
      hiddenStates,
      attentionMask: attentionMask,
      context: context,
    );
    hiddenStates = residual + hiddenStates;

    residual = hiddenStates;
    hiddenStates = layerNorm2.forward(hiddenStates, context: context);
    hiddenStates = mlp.forward(hiddenStates, context: context);
    hiddenStates = residual + hiddenStates;
    return hiddenStates;
  }

  @override
  void resetParameters() {
    selfAttention.resetParameters();
    layerNorm1.resetParameters();
    mlp.resetParameters();
    layerNorm2.resetParameters();
  }

  @override
  final Map<String, dynamic> meta = const {};

  @override
  final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [
    selfAttention,
    layerNorm1,
    mlp,
    layerNorm2,
  ];

  static Future<ClipEncoderLayer> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required ClipTextConfig config,
    String prefix = '',
    ClipEncoderLayerNames names = const ClipEncoderLayerNames(),
    required String name,
  }) async {
    final selfAttention = await ClipAttention.loadFromSafeTensor(
      loader,
      prefix: '$prefix${names.selfAttention}.',
      name: names.selfAttention,
      config: config,
    );
    final layerNorm1 = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix${names.layerNorm1}.',
      name: names.layerNorm1,
      normalizedShape: [config.embedDim],
      eps: config.layerNormEps,
    );
    final layerNorm2 = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '$prefix${names.layerNorm2}.',
      name: names.layerNorm2,
      normalizedShape: [config.embedDim],
      eps: config.layerNormEps,
    );
    final mlp = await ClipMlp.loadFromSafeTensor(
      loader,
      prefix: '$prefix${names.mlp}.',
      name: names.mlp,
      config: config,
    );
    return ClipEncoderLayer(
      name: name,
      selfAttention: selfAttention,
      layerNorm1: layerNorm1,
      mlp: mlp,
      layerNorm2: layerNorm2,
    );
  }
}

class ClipEncoderLayerNames {
  final String selfAttention;
  final String layerNorm1;
  final String mlp;
  final String layerNorm2;

  const ClipEncoderLayerNames({
    this.selfAttention = 'self_attn',
    this.layerNorm1 = 'layer_norm1',
    this.mlp = 'mlp',
    this.layerNorm2 = 'layer_norm2',
  });
}

class ClipMlp extends Module implements SimpleModule {
  final LinearLayer linear1;
  final LinearLayer linear2;
  final Activation activation;

  ClipMlp({
    required super.name,
    required this.linear1,
    required this.linear2,
    required this.activation,
  });

  @override
  Tensor forward(Tensor hiddenStates, {required Context context}) {
    hiddenStates = linear1.forward(hiddenStates, context: context);
    hiddenStates = activation.forward(hiddenStates, context: context);
    hiddenStates = linear2.forward(hiddenStates, context: context);
    return hiddenStates;
  }

  @override
  void resetParameters() {
    linear1.resetParameters();
    linear2.resetParameters();
  }

  @override
  late final Map<String, dynamic> meta = {
    'linear1': linear1.meta,
    'linear2': linear2.meta,
    'activation': activation.runtimeType.toString(),
  };

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Module> submodules = [linear1, linear2];

  static Future<ClipMlp> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required ClipTextConfig config,
    String prefix = '',
    required String name,
  }) async {
    final linear1 = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}fc1.',
    );
    final linear2 = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}fc2.',
    );
    // TODO verify shape against config
    return ClipMlp(
      name: name,
      linear1: linear1,
      linear2: linear2,
      activation: config.activation,
    );
  }
}

class ClipTextEmbeddings extends Module implements SimpleModule {
  final EmbeddingLayer tokenEmbedding;
  final EmbeddingLayer positionEmbedding;
  final Tensor positionIds;

  ClipTextEmbeddings({
    required super.name,
    required this.tokenEmbedding,
    required this.positionEmbedding,
    required this.positionIds,
  });

  @override
  Tensor forward(
    Tensor inputIds, {
    Tensor? positionIds,
    required Context context,
  }) {
    final seqLength = inputIds.shape.last;
    if (seqLength > maxPositionEmbeddings) {
      throw ArgumentError.value(
        'Input sequence length is greater than the maximum position embeddings',
      );
    }
    positionIds ??= this.positionIds.expand([-1, seqLength]);
    final positionEmbeddings = positionEmbedding.forward(
      positionIds,
      context: context,
    );
    return tokenEmbedding.forward(inputIds, context: context) +
        positionEmbeddings;
  }

  int get embeddingDim => tokenEmbedding.embeddingDim;

  int get vocabSize => tokenEmbedding.numEmbeddings;

  int get maxPositionEmbeddings => positionEmbedding.numEmbeddings;

  @override
  void resetParameters() {
    tokenEmbedding.resetParameters();
    positionEmbedding.resetParameters();
  }

  @override
  late final Map<String, dynamic> meta = {
    'tokenEmbedding': tokenEmbedding.meta,
    'positionEmbedding': positionEmbedding.meta,
  };

  @override
  late final Iterable<Module> submodules = [tokenEmbedding, positionEmbedding];

  @override
  late final Iterable<Tensor> parameters = [positionIds];

  static Future<ClipTextEmbeddings> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = 'text_model.embeddings.',
    required String name,
    final String tokenEmbeddingName = 'token_embedding',
    final String positionEmbeddingName = 'position_embedding',
    final String positionIdsName = 'position_ids',
  }) async {
    // final tokenEmbedding = loader.loadByName(prefix + 'token_embedding.weight');
    final tokenEmbedding = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      name: tokenEmbeddingName,
      prefix: '$prefix$tokenEmbeddingName.',
    );
    final positionEmbedding = await EmbeddingLayer.loadFromSafeTensor(
      loader,
      name: positionEmbeddingName,
      prefix: '$prefix$positionEmbeddingName.',
    );
    final Tensor positionIds = Tensor.arange(
      0,
      positionEmbedding.numEmbeddings,
      name: positionIdsName,
    ).expand([1, -1]);

    return ClipTextEmbeddings(
      name: name,
      tokenEmbedding: tokenEmbedding,
      positionEmbedding: positionEmbedding,
      positionIds: positionIds,
    );
  }
}
