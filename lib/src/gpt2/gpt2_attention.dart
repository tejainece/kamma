import 'dart:math';
import 'package:kamma/kamma.dart';

class GPT2Attention extends Module {
  final int embedDim;
  final int numHeads;
  final int headDim;
  final int splitSize;
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
    required GPT2Config config,
    this.isCrossAttention = false,
    this.layerIdx = 0,
    required this.cAttn,
    required this.cProj,
    required this.attnDropout,
    required this.residDropout,
  }) : embedDim = config.nEmbd,
       numHeads = config.nHead,
       headDim = config.nEmbd ~/ config.nHead,
       splitSize = config.nEmbd,
       scaleAttnWeights = config.scaleAttnWeights,
       scaleAttnByInverseLayerIdx = config.scaleAttnByInverseLayerIdx,
       reorderAndUpcastAttn = config.reorderAndUpcastAttn {
    if (embedDim % numHeads != 0) {
      throw ArgumentError(
        "embed_dim must be divisible by num_heads (got `embed_dim`: $embedDim"
        " and `num_heads`: $numHeads).",
      );
    }
  }

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
  late final Iterable<Tensor> parameters = [
    ...cAttn.parameters,
    ...cProj.parameters,
  ];

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
    required GPT2Config config,
    required String name,
    bool isCrossAttention = false,
    int layerIdx = 0,
  }) {
    final cAttn = LinearLayer.make(
      name: 'c_attn',
      inFeatures: config.nEmbd,
      outFeatures: 3 * config.nEmbd,
    );

    final cProj = LinearLayer.make(
      name: 'c_proj',
      inFeatures: config.nEmbd,
      outFeatures: config.nEmbd,
    );

    final attnDropout = Dropout(config.attnPdrop);
    final residDropout = Dropout(config.residPdrop);

    return GPT2Attention(
      name: name,
      config: config,
      isCrossAttention: isCrossAttention,
      layerIdx: layerIdx,
      cAttn: cAttn,
      cProj: cProj,
      attnDropout: attnDropout,
      residDropout: residDropout,
    );
  }

  Future<void> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
  }) async {
    // Load c_attn
    if (loader.hasTensor('${prefix}c_attn.weight')) {
      final weight = await loader.loadByName('${prefix}c_attn.weight');
      cAttn.weight.copy_(weight.transpose(0, 1));
    }
    if (loader.hasTensor('${prefix}c_attn.bias')) {
      final bias = await loader.loadByName('${prefix}c_attn.bias');
      cAttn.bias!.copy_(bias);
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
