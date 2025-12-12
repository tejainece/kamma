import 'dart:math';
import 'package:kamma/kamma.dart';

class GPT2Attention extends Module {
  final int numHeads;
  final bool scaleAttnWeights;
  final bool scaleAttnByInverseLayerIdx;
  // TODO use cache instead
  final bool reorderAndUpcastAttn;
  final bool isCrossAttention;
  final int layerIdx;

  /// Contains/performs q_proj, k_proj, v_proj in single linear layer for efficiency.
  final LinearLayer cAttn;
  final LinearLayer cProj;
  final Dropout attnDropout;
  final Dropout residDropout;

  final Tensor bias;

  GPT2Attention({
    required super.name,
    this.isCrossAttention = false,
    this.layerIdx = 0,
    required this.cAttn,
    required this.cProj,
    required this.attnDropout,
    required this.residDropout,
    required this.bias,
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

  // TODO use SDAP
  ({Tensor attentionOutput, Tensor attentionWeights}) _attn(
    Tensor query,
    Tensor key,
    Tensor value, {
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  }) {
    final [batchSize, numHeads, qSeqLength, headDim] = query.shape;
    final [_, _, kSeqLength, _] = key.shape;

    // Preallocate attention weights tensor. Will be populated by baddbmm
    Tensor attentionWeights = Tensor.empty(
      [batchSize * numHeads, qSeqLength, kSeqLength],
      datatype: DataType.float32,
      device: context.device,
    );

    // Scale factor for attention weights as descibed in attention formula
    double scaleFactor = 1.0;
    if (scaleAttnWeights) {
      scaleFactor /= sqrt(value.shape.last);
    }
    if (scaleAttnByInverseLayerIdx) {
      scaleFactor /= (layerIdx + 1);
    }

    context.device.withAutocast(false, () {
      Tensor q = query
          .reshape([-1, qSeqLength, headDim])
          .to(dataType: DataType.float32);
      Tensor k = key
          .transpose(-1, -2)
          .reshape([-1, headDim, kSeqLength])
          .to(dataType: DataType.float32);
      attentionWeights.baddbmm_(q, k, beta: 0, alpha: scaleFactor);
      attentionWeights = attentionWeights.reshape([
        batchSize,
        numHeads,
        qSeqLength,
        kSeqLength,
      ]);
    });

    // Apply causal mask. Cross-attention is only performed on encoder-decoder transformer.
    // For encoder-decoder transformer, causal mask is not applied.
    if (!isCrossAttention) {
      final causalMask = bias.index([
        .all,
        .all,
        // Mask for query
        // During inference of decoder-only transformer, only the last token is processed (qSeqLength = 1).
        // During training of decoder-only transformer, qSeqLength = kSeqLength.
        .slice(kSeqLength - qSeqLength, kSeqLength),
        // Mask for key
        .to(kSeqLength),
      ]);
      attentionWeights = causalMask.where(
        attentionWeights,
        attentionWeights.dataType.fInfo.min,
      );
    }

    if (reorderAndUpcastAttn) {
      // TODO: Implement reorder and upcast if needed, usually for mixed precision
    }

    if (attentionMask != null) {
      attentionWeights = attentionWeights + attentionMask;
    }

    attentionWeights = attentionWeights.softmax(-1);

    if (!attentionWeights.isFloat32) {
      throw Exception(
        "Error upcasting! Expected float32, got ${attentionWeights.dataType}",
      );
    }
    attentionWeights = attentionWeights.to(dataType: value.dataType);
    attentionWeights = attnDropout.forward(attentionWeights, context: context);

    if (headMask != null) {
      attentionWeights = attentionWeights * headMask;
    }

    Tensor attentionOutput = attentionWeights.matmul(value);
    attentionOutput = attentionOutput.transpose(1, 2);

    return (
      attentionOutput: attentionOutput,
      attentionWeights: attentionWeights,
    );
  }

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

  /// [input] is of size (batch, seq_length, embed_dim)
  ({Tensor attentionOutput, Tensor attentionWeights}) forward(
    Tensor input, {
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

    // TODO key,value cache

    Tensor query, key, value;
    if (isCrossAttention) {
      assert(
        encoderHiddenStates != null,
        "encoder_hidden_states must be provided for cross attention",
      );
      query = cAttn.forward(input, context: context);
      query = _splitHeads(query, numHeads, headDim);

      final keyVal = cAttn.forward(encoderHiddenStates!, context: context);
      final splitKeyVal = keyVal.splitEqually(splitSize, dim: 2);
      key = _splitHeads(splitKeyVal[0], numHeads, headDim);
      value = _splitHeads(splitKeyVal[1], numHeads, headDim);
    } else {
      final qkv = cAttn.forward(input, context: context);
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

    var (:attentionOutput, :attentionWeights) = _attn(
      query,
      key,
      value,
      attentionMask: attentionMask,
      headMask: headMask,
      context: context,
    );

    attentionOutput = _mergeHeads(attentionOutput, numHeads, headDim);
    attentionOutput = cProj.forward(attentionOutput, context: context);
    attentionOutput = residDropout.forward(attentionOutput, context: context);

    // TODO: Return present and attentions if needed
    return (
      attentionOutput: attentionOutput,
      attentionWeights: attentionWeights,
    );
  }

  @override
  void resetParameters() {
    cAttn.resetParameters();
    cProj.resetParameters();
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Tensor> nonTrainableParameters = [bias];

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
    required int maxPositionEmbeddings,
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

    final bias = Tensor.ones(
      [maxPositionEmbeddings, maxPositionEmbeddings],
      dataType: DataType.boolean,
    ).tril().view([1, 1, maxPositionEmbeddings, maxPositionEmbeddings]);

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
      bias: bias,
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
    required int maxPositionEmbeddings,
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

    final bias = Tensor.ones(
      [maxPositionEmbeddings, maxPositionEmbeddings],
      dataType: DataType.boolean,
    ).tril().view([1, 1, maxPositionEmbeddings, maxPositionEmbeddings]);

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
      bias: bias,
    );
  }
}
