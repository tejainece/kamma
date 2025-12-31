import 'dart:math';
import 'package:tensor/tensor.dart';
import 'attention_methods.dart';

class GPT2Attention extends Module {
  final int layerIdx;
  final int numHeads;

  /// Contains/performs q_proj, k_proj, v_proj in single linear layer for efficiency.
  /// Only used if [isCrossAttention] is false, because in cross-attention, key and
  /// value come from the encoder.
  final LinearTransposed qkvAttention;
  final LinearTransposed outputProjection;
  final Dropout attentionDropout;
  final Dropout residualDropout;
  late AttentionCache keyValueCache;
  late final GPT2AttentionMethod attentionMethod;

  final bool scaleAttnByInverseLayerIdx;
  final bool isCrossAttention;

  GPT2Attention({
    required super.name,
    required this.layerIdx,
    required this.numHeads,
    required this.qkvAttention,
    required this.outputProjection,
    required this.attentionDropout,
    required this.residualDropout,
    required int maxPositionEmbeddings,
    required GPT2AttentionMethodType attentionMethod,
    this.isCrossAttention = false,
    required this.scaleAttnByInverseLayerIdx,
  }) {
    if (embedDim % numHeads != 0) {
      throw ArgumentError(
        "embed_dim must be divisible by num_heads (got `embed_dim`: $embedDim"
        " and `num_heads`: $numHeads).",
      );
    }

    double scaleFactor = 1.0 / sqrt(headDim);

    if (scaleAttnByInverseLayerIdx) {
      scaleFactor /= layerIdx + 1;
    }

    this.attentionMethod = GPT2AttentionMethod.make(
      attentionMethod,
      scaleFactor: scaleFactor,
      isCausal: !isCrossAttention,
      attnDropout: attentionDropout,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );

    if (attentionMethod != GPT2AttentionMethodType.pagedAttention) {
      keyValueCache = AttentionCache.empty();
    } else {
      // TODO intialize cache for paged attention
      throw UnimplementedError("Paged Attention is not implemented yet.");
    }
  }

  int get embedDim => qkvAttention.numInFeatures;

  int get splitSize => embedDim;

  late final int headDim = embedDim ~/ numHeads;

  /// [tensor] is of shape (batch, seq_length, embed_dim). embed_dim consists of multiple heads([numHeads])
  /// each of size [attnHeadSize]. This function splits the [tensor] into multiple heads with each head
  /// of shape (seq_length, attnHeadSize).
  /// It returns a tensor of shape (batch, numHeads, seq_length, attnHeadSize).
  Tensor _splitHeads(Tensor tensor, int numHeads, int attnHeadSize) {
    final newShape = [
      ...tensor.shape.sublist(0, tensor.shape.length - 1),
      numHeads,
      attnHeadSize,
    ];
    tensor = tensor.view(newShape);
    return tensor.permute([
      0,
      2,
      1,
      3,
    ]); // (batch, head, seq_length, head_features)
  }

  Tensor _mergeHeads(Tensor tensor, int numHeads, int attnHeadSize) {
    tensor = tensor.permute([0, 2, 1, 3]).contiguous();
    final newShape = [
      ...tensor.shape.sublist(0, tensor.shape.length - 2),
      numHeads * attnHeadSize,
    ];
    return tensor.view(newShape);
  }

  /// [inputEmbeddings] is of size (batch, seq_length, embed_dim)
  /// [attentionMask] is used to mask out certain positions in the sequence. For example, to mask
  /// out padding tokens. It is of shape (batch, 1, 1, seq_length).
  /// [headMask] is used to mask out certain heads in the attention.
  Tensor forward(
    Tensor inputEmbeddings, {
    // TODO implement cachePosition
    Tensor? attentionMask,
    Tensor? headMask,
    Tensor? encoderHiddenStates,
    List<Tensor>? outputAttentions,
    required Context context,
  }) {
    context.onloadModule(this);

    // TODO key,value cache
    if (!context.isTraining) {
      // TODO
    }

    Tensor query, key, value;
    if (isCrossAttention) {
      assert(
        encoderHiddenStates != null,
        "encoder_hidden_states must be provided for cross attention",
      );
      query = qkvAttention.forward(inputEmbeddings, context: context);
      query = _splitHeads(query, numHeads, headDim);

      // TODO use keyValueCache
      final keyVal = qkvAttention.forward(
        encoderHiddenStates!,
        context: context,
      );
      final splitKeyVal = keyVal.splitEqually(splitSize, dim: 2);
      key = _splitHeads(splitKeyVal[0], numHeads, headDim);
      value = _splitHeads(splitKeyVal[1], numHeads, headDim);
    } else {
      final qkv = qkvAttention.forward(inputEmbeddings, context: context);
      final splitQkv = qkv.splitEqually(splitSize, dim: 2);
      query = _splitHeads(splitQkv[0], numHeads, headDim);
      key = _splitHeads(splitQkv[1], numHeads, headDim);
      value = _splitHeads(splitQkv[2], numHeads, headDim);
    }

    if (!context.isTraining) {
      // TODO implement cache_position (updating specific position in cache instead of at the end)
      keyValueCache.update(newKey: key, newValue: value);
      key = keyValueCache.key;
      value = keyValueCache.value;
    }

    var (:attentionOutput, :attentionWeights) = attentionMethod.perform(
      query,
      key,
      value,
      attentionMask: attentionMask,
      headMask: headMask,
      context: context,
    );

    attentionOutput = _mergeHeads(attentionOutput, numHeads, headDim);
    attentionOutput = outputProjection.forward(
      attentionOutput,
      context: context,
    );
    attentionOutput = residualDropout.forward(
      attentionOutput,
      context: context,
    );

    outputAttentions?.add(attentionWeights);
    return attentionOutput;
  }

  @override
  void resetParameters() {
    qkvAttention.resetParameters();
    outputProjection.resetParameters();
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Tensor> nonTrainableParameters = [];

  @override
  late final Iterable<Module> submodules = [
    qkvAttention,
    outputProjection,
    attentionDropout,
    residualDropout,
    attentionMethod,
  ];

  @override
  Map<String, dynamic> get meta => {
    "embedDim": embedDim,
    "numHeads": numHeads,
    "headDim": headDim,
    "splitSize": splitSize,
    "scaleAttnByInverseLayerIdx": scaleAttnByInverseLayerIdx,
    "isCrossAttention": isCrossAttention,
    "layerIdx": layerIdx,
  };

  static GPT2Attention make({
    required String name,
    required bool isCrossAttention,
    required int layerIdx,
    required int embedDim,
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required int numHeads,
    required bool scaleAttnByInverseLayerIdx,
    required int maxPositionEmbeddings,
    GPT2AttentionMethodType attnFuncType = .eager,
    String qkvAttentionName = 'c_attn',
    String outputProjectionName = 'c_proj',
  }) {
    final qkvAttention = LinearTransposed.make(
      name: qkvAttentionName,
      numInFeatures: embedDim,
      numOutFeatures: 3 * embedDim,
    );

    final outputProjection = LinearTransposed.make(
      name: outputProjectionName,
      numInFeatures: embedDim,
      numOutFeatures: embedDim,
    );

    final attnDropout = Dropout(attentionDropoutProbability);
    final residDropout = Dropout(residualDropoutProbability);

    return GPT2Attention(
      name: name,
      isCrossAttention: isCrossAttention,
      layerIdx: layerIdx,
      qkvAttention: qkvAttention,
      outputProjection: outputProjection,
      attentionDropout: attnDropout,
      residualDropout: residDropout,
      numHeads: numHeads,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      maxPositionEmbeddings: maxPositionEmbeddings,
      attentionMethod: attnFuncType,
    );
  }

  static Future<GPT2Attention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int layerIdx,
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    bool isCrossAttention = false,
    required int numHeads,
    String qkvAttentionName = 'c_attn',
    String outputProjectionName = 'c_proj',
    required bool scaleAttnByInverseLayerIdx,
    GPT2AttentionMethodType attnFuncType = .eager,
    required int maxPositionEmbeddings,
  }) async {
    final qkvAttention = await LinearTransposed.loadFromSafeTensor(
      loader,
      prefix: '$prefix$qkvAttentionName.',
      name: qkvAttentionName,
    );
    final outputProjection = await LinearTransposed.loadFromSafeTensor(
      loader,
      prefix: '$prefix$outputProjectionName.',
      name: outputProjectionName,
    );

    final attnDropout = Dropout(attentionDropoutProbability);
    final residDropout = Dropout(residualDropoutProbability);

    return GPT2Attention(
      name: name,
      isCrossAttention: isCrossAttention,
      layerIdx: layerIdx,
      qkvAttention: qkvAttention,
      outputProjection: outputProjection,
      attentionDropout: attnDropout,
      residualDropout: residDropout,
      numHeads: numHeads,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      attentionMethod: attnFuncType,
      maxPositionEmbeddings: maxPositionEmbeddings,
    );
  }
}

// TODO support offloading and onloading
class AttentionCache {
  /// key and value are of shape [batchSize, numHeads, seqLength, headDim]
  Tensor _key;
  Tensor _value;

  AttentionCache({required Tensor key, required Tensor value})
    : _key = key,
      _value = value;

  AttentionCache.empty()
    : _key = Tensor.empty([0, 0, 0, 0]),
      _value = Tensor.empty([0, 0, 0, 0]);

  /// [newKey] and [newValue] are of shape [batchSize, numHeads, seqLength, headDim]
  void update({required Tensor newKey, required Tensor newValue}) {
    if (_key.numel == 0) {
      _key = newKey;
      _value = newValue;
    } else {
      _key = Tensor.cat([_key, newKey], dim: -2);
      _value = Tensor.cat([_value, newValue], dim: -2);
    }
  }

  Tensor get key => _key;

  Tensor get value => _value;

  int get seqLength => _key.shape[2];

  void reset({Tensor? key, Tensor? value}) {
    _key = key ?? Tensor.empty([0, 0, 0, 0]);
    _value = value ?? Tensor.empty([0, 0, 0, 0]);
  }
}
