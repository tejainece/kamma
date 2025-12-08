import 'dart:math';
import 'package:kamma/kamma.dart';

class GPT2Attention extends Module {
  final int numHeads;
  final bool scaleAttnWeights;
  final bool scaleAttnByInverseLayerIdx;
  final bool reorderAndUpcastAttn;
  final bool isCrossAttention;
  final int layerIdx;

  final LinearLayer cAttn;
  final LinearLayer cProj;
  final Dropout attnDropout;
  final Dropout residDropout;

  GPT2Attention({
    required super.name,
    this.isCrossAttention = false,
    this.layerIdx = 0,
    required this.cAttn,
    required this.cProj,
    required this.attnDropout,
    required this.residDropout,
    required this.numHeads,
    required this.scaleAttnWeights,
    required this.scaleAttnByInverseLayerIdx,
    required this.reorderAndUpcastAttn,
  }) {
    if (embedDim % numHeads != 0) {
      throw ArgumentError(
        "embed_dim must be divisible by num_heads (got `embed_dim`: $embedDim"
        " and `num_heads`: $numHeads).",
      );
    }
  }

  int get embedDim => cAttn.inFeatures;

  int get splitSize => embedDim;

  late final int headDim = embedDim ~/ numHeads;

  Tensor _attn(
    Tensor query,
    Tensor key,
    Tensor value, {
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  }) {
    Tensor attnWeights = query.matmul(key.transpose(-1, -2));

    if (scaleAttnWeights) {
      attnWeights = attnWeights / sqrt(value.shape.last.toDouble());
    }

    if (scaleAttnByInverseLayerIdx) {
      attnWeights = attnWeights / (layerIdx + 1).toDouble();
    }

    if (reorderAndUpcastAttn) {
      // TODO: Implement reorder and upcast if needed, usually for mixed precision
    }

    if (attentionMask != null) {
      attnWeights = attnWeights + attentionMask;
    }

    attnWeights = attnWeights.softmax(-1);
    attnWeights = attnDropout.forward(attnWeights, context: context);

    if (headMask != null) {
      attnWeights = attnWeights * headMask;
    }

    Tensor attnOutput = attnWeights.matmul(value);
    return attnOutput;
  }

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

  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? layerPast,
    Tensor? attentionMask,
    Tensor? headMask,
    Tensor? encoderHiddenStates,
    Tensor? encoderAttentionMask,
    bool useCache = false,
    bool outputAttentions = false,
    required Context context,
  }) {
    context.onloadModule(this);

    Tensor query, key, value;
    if (isCrossAttention) {
      assert(
        encoderHiddenStates != null,
        "encoder_hidden_states must be provided for cross attention",
      );
      query = cAttn.forward(hiddenStates, context: context);
      query = _splitHeads(query, numHeads, headDim);

      final keyVal = cAttn.forward(encoderHiddenStates!, context: context);
      final splitKeyVal = keyVal.splitEqually(splitSize, dim: 2);
      key = _splitHeads(splitKeyVal[0], numHeads, headDim);
      value = _splitHeads(splitKeyVal[1], numHeads, headDim);
    } else {
      final qkv = cAttn.forward(hiddenStates, context: context);
      final splitQkv = qkv.splitEqually(splitSize, dim: 2);
      query = _splitHeads(splitQkv[0], numHeads, headDim);
      key = _splitHeads(splitQkv[1], numHeads, headDim);
      value = _splitHeads(splitQkv[2], numHeads, headDim);
    }

    if (layerPast != null) {
      final pastKey = layerPast[0];
      final pastValue = layerPast[1];
      key = Tensor.cat([pastKey, key], dim: -2);
      value = Tensor.cat([pastValue, value], dim: -2);
    }

    Tensor? present;
    if (useCache) {
      present = Tensor.cat([key.unsqueeze(0), value.unsqueeze(0)], dim: 0);
    }

    Tensor attnOutput = _attn(
      query,
      key,
      value,
      attentionMask: attentionMask,
      headMask: headMask,
      context: context,
    );

    attnOutput = _mergeHeads(attnOutput, numHeads, headDim);
    attnOutput = cProj.forward(attnOutput, context: context);
    attnOutput = residDropout.forward(attnOutput, context: context);

    // TODO: Return present and attentions if needed
    return attnOutput;
  }

  @override
  void resetParameters() {
    cAttn.resetParameters();
    cProj.resetParameters();
    // Dropouts don't need reset
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [
    cAttn,
    cProj,
    attnDropout,
    residDropout,
  ];

  @override
  Map<String, dynamic> get meta => {
    "embedDim": embedDim,
    "numHeads": numHeads,
    "headDim": headDim,
    "splitSize": splitSize,
    "scaleAttnWeights": scaleAttnWeights,
    "scaleAttnByInverseLayerIdx": scaleAttnByInverseLayerIdx,
    "reorderAndUpcastAttn": reorderAndUpcastAttn,
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
    required bool scaleAttnWeights,
    required bool scaleAttnByInverseLayerIdx,
    required bool reorderAndUpcastAttn,
  }) {
    final cAttn = LinearLayer.make(
      name: 'c_attn',
      inFeatures: embedDim,
      outFeatures: 3 * embedDim,
    );

    final cProj = LinearLayer.make(
      name: 'c_proj',
      inFeatures: embedDim,
      outFeatures: embedDim,
    );

    final attnDropout = Dropout(attentionDropoutProbability);
    final residDropout = Dropout(residualDropoutProbability);

    return GPT2Attention(
      name: name,
      isCrossAttention: isCrossAttention,
      layerIdx: layerIdx,
      cAttn: cAttn,
      cProj: cProj,
      attnDropout: attnDropout,
      residDropout: residDropout,
      numHeads: numHeads,
      scaleAttnWeights: scaleAttnWeights,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      reorderAndUpcastAttn: reorderAndUpcastAttn,
    );
  }

  static Future<GPT2Attention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int layerIdx,
    required double attentionDropoutProbability,
    required double residualDropoutProbability,
    required bool isCrossAttention,
    required int numHeads,
    String cAttnName = 'c_attn',
    String cProjName = 'c_proj',
    required bool scaleAttnWeights,
    required bool scaleAttnByInverseLayerIdx,
    required bool reorderAndUpcastAttn,
  }) async {
    final cAttn = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '$prefix$cAttnName.',
      name: cAttnName,
    );
    final cProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '$prefix$cProjName.',
      name: cProjName,
    );

    final attnDropout = Dropout(attentionDropoutProbability);
    final residDropout = Dropout(residualDropoutProbability);

    return GPT2Attention(
      name: name,
      isCrossAttention: isCrossAttention,
      layerIdx: layerIdx,
      cAttn: cAttn,
      cProj: cProj,
      attnDropout: attnDropout,
      residDropout: residDropout,
      numHeads: numHeads,
      scaleAttnWeights: scaleAttnWeights,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      reorderAndUpcastAttn: reorderAndUpcastAttn,
    );
  }
}
