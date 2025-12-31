import 'package:kamma/kamma.dart';

enum GPT2AttentionMethodType {
  eagerUpscale,
  eager,
  sdap,
  flashAttention2,
  flashAttention3,
  flexAttention,
  pagedAttention,
  sdpaPaged,
  eagerPaged,
}

// TODO see if this can be reused for other models
abstract class GPT2AttentionMethod implements Module {
  GPT2AttentionMethodType get type;

  double get scaleFactor;

  bool get isCausal;

  Dropout get attnDropout;

  ({Tensor attentionOutput, Tensor attentionWeights}) perform(
    Tensor query,
    Tensor key,
    Tensor value, {

    /// [attentionMask] is used to mask out certain positions in the sequence. For example, to mask
    /// out padding tokens. It is of shape (batch, 1, 1, seq_length).
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  });

  /// [createCausalMask] uses this method to compute [GPT2AttentionMethodType] specific mask.
  CausalMask? makeCausalMask(
    int batchSize,
    int qLen,
    DataType dataType,
    Device device, {
    int pastKeyValuesLength = 0,
    Tensor? attentionMask,
  });

  static GPT2AttentionMethod make(
    GPT2AttentionMethodType type, {
    required double scaleFactor,
    required bool isCausal,
    required int maxPositionEmbeddings,
    required Dropout attnDropout,
  }) {
    switch (type) {
      case GPT2AttentionMethodType.eagerUpscale:
        return GPT2AttentionMethodEagerUpscale(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.sdap:
        return GPT2AttentionMethodSDAP(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.flashAttention2:
        return GPT2AttentionMethodFlashAttention2(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.flashAttention3:
        return GPT2AttentionFuncFlashAttention3(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.flexAttention:
        return GPT2AttentionFuncFlexAttention(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.pagedAttention:
        return GPT2AttentionMethodPagedAttention(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.sdpaPaged:
        return GPT2AttentionFuncSdpaPaged(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.eagerPaged:
        return GPT2AttentionFuncEagerPaged(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      case GPT2AttentionMethodType.eager:
        return GPT2AttentionMethodEager(
          scaleFactor: scaleFactor,
          isCausal: isCausal,
          maxPositionEmbeddings: maxPositionEmbeddings,
          attnDropout: attnDropout,
        );
      default:
        throw Exception('Unknown GPT2AttentionFuncType: $type');
    }
  }
}

class GPT2AttentionMethodEagerUpscale extends Module
    implements GPT2AttentionMethod {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.eagerUpscale;

  @override
  final double scaleFactor;

  @override
  final bool isCausal;

  /// Used to compute causal mask
  final Tensor premadeCausalMask;

  @override
  final Dropout attnDropout;

  GPT2AttentionMethodEagerUpscale({
    required this.scaleFactor,
    required this.isCausal,
    required this.attnDropout,
    required int maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_eager',
  }) : premadeCausalMask = Tensor.ones(
         [maxPositionEmbeddings, maxPositionEmbeddings],
         dataType: DataType.boolean,
       ).tril().view([1, 1, maxPositionEmbeddings, maxPositionEmbeddings]);

  @override
  /// Create a 4D float mask of shape `(batch_size, 1, query_length, kv_length)` where a value of 0 indicates that
  /// the element should take part in the attention computation, and -inf (minimum value for the given `dtype`) that
  /// it should not.
  ///
  /// TODO document when [attentionMask] is provided
  /// TODO document when [attentionMask] is used
  CausalMask? makeCausalMask(
    int batchSize,

    /// Number of new tokens to process.
    int qLen,
    DataType dataType,
    Device device, {
    int pastKeyValuesLength = 0,
    Tensor? attentionMask,
  }) {
    final minVal = dataType.fInfo.min;
    final targetLength = qLen;
    final kvLen = qLen + pastKeyValuesLength;

    final rangeRow = Tensor.arange(
      0,
      targetLength,
      device: device,
    ).view([targetLength, 1]);
    final rangeCol = Tensor.arange(0, kvLen, device: device).view([1, kvLen]);

    final maskCondition = rangeCol.gt(rangeRow + pastKeyValuesLength);

    Tensor causalMask = Tensor.zeros(
      [targetLength, kvLen],
      device: device,
      dataType: dataType,
    );

    causalMask = causalMask.maskedFill(maskCondition, minVal);
    causalMask = causalMask.view([1, 1, targetLength, kvLen]);

    // Match transformers behavior: expand to batch size
    if (batchSize > 1) {
      causalMask = causalMask.expand([batchSize, 1, targetLength, kvLen]);
    }

    // 2. Combine with Attention Mask if provided
    if (attentionMask != null) {
      if (attentionMask.shape.length == 2) {
        Tensor extAttentionMask = attentionMask.view([batchSize, 1, 1, kvLen]);
        extAttentionMask = extAttentionMask.to(dataType: dataType);
        extAttentionMask =
            (Tensor.ones(
                  extAttentionMask.shape,
                  device: device,
                  dataType: dataType,
                ) -
                extAttentionMask) *
            minVal;
        causalMask = causalMask + extAttentionMask;
      } else if (attentionMask.shape.length == 4) {
        // If it is already 4D, we assume it is prepared
        causalMask = causalMask + attentionMask;
      }
    }

    return SimpleCausalMask(causalMask);
  }

  @override
  ({Tensor attentionOutput, Tensor attentionWeights}) perform(
    Tensor query,
    Tensor key,
    Tensor value, {

    /// [attentionMask] is used to mask out certain positions in the sequence. For example, to mask
    /// out padding tokens. It is of shape (batch, 1, 1, seq_length).
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  }) {
    context.onloadModule(this);

    final [batchSize, numHeads, qSeqLength, headDim] = query.shape;
    final [_, _, kSeqLength, _] = key.shape;

    // Preallocate attention weights tensor. Will be populated by baddbmm
    Tensor attentionWeights = Tensor.empty(
      [batchSize * numHeads, qSeqLength, kSeqLength],
      datatype: DataType.float32,
      device: context.device,
    );

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
    if (isCausal) {
      final causalMask = premadeCausalMask.index([
        // Calculate for all batches
        .all,
        // Calculate for all heads
        .all,
        // Mask for query
        // During inference of decoder-only transformer, only the last token is processed (qSeqLength = 1).
        // During training of decoder-only transformer, qSeqLength = kSeqLength.
        .slice(kSeqLength - qSeqLength, kSeqLength),
        // Mask for key
        .to(kSeqLength),
      ]);
      // mask is 1 where we want to keep, 0 where we want to mask
      attentionWeights = attentionWeights.maskedFill(
        causalMask.eq(0),
        attentionWeights.dataType.fInfo.min,
      );
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

  @override
  final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Tensor> nonTrainableParameters = [premadeCausalMask];

  @override
  final Iterable<Module> submodules = [];

  @override
  final Map<String, Tensor> meta = {};

  @override
  void resetParameters() {}
}

class GPT2AttentionMethodEager extends Module implements GPT2AttentionMethod {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.eager;

  @override
  final double scaleFactor;

  @override
  final bool isCausal;

  /// Used to compute causal mask
  final Tensor premadeCausalMask;

  @override
  final Dropout attnDropout;

  GPT2AttentionMethodEager({
    required this.scaleFactor,
    required this.isCausal,
    required this.attnDropout,
    required int maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_eager',
  }) : premadeCausalMask = Tensor.ones(
         [maxPositionEmbeddings, maxPositionEmbeddings],
         dataType: DataType.boolean,
       ).tril().view([1, 1, maxPositionEmbeddings, maxPositionEmbeddings]);

  @override
  CausalMask? makeCausalMask(
    int batchSize,
    int qLen,
    DataType dataType,
    Device device, {
    int pastKeyValuesLength = 0,
    Tensor? attentionMask,
  }) {
    // Eager Attention also uses float mask
    final minVal = dataType.fInfo.min;
    final sourceLength = qLen + pastKeyValuesLength;

    // 1. Create base causal mask
    final rangeRow = Tensor.arange(
      0,
      qLen,
      device: device,
      dataType: DataType.float32,
    ).view([qLen, 1]);
    final rangeCol = Tensor.arange(
      0,
      sourceLength,
      device: device,
      dataType: DataType.float32,
    ).view([1, sourceLength]);

    // mask logic: j > i + pastKeyValuesLength => mask
    final maskCondition = rangeCol.gt(
      rangeRow + pastKeyValuesLength.toDouble(),
    );

    Tensor causalMask = Tensor.zeros(
      [qLen, sourceLength],
      device: device,
      dataType: dataType,
    );

    causalMask = causalMask.maskedFill(maskCondition, minVal);
    causalMask = causalMask.view([1, 1, qLen, sourceLength]);

    // Match transformers behavior: expand to batch size
    if (batchSize > 1) {
      causalMask = causalMask.expand([batchSize, 1, qLen, sourceLength]);
    }

    // 2. Combine with Attention Mask if provided
    if (attentionMask != null) {
      if (attentionMask.shape.length == 2) {
        Tensor extAttentionMask = attentionMask.view([
          batchSize,
          1,
          1,
          sourceLength,
        ]);
        extAttentionMask = extAttentionMask.to(dataType: dataType);
        extAttentionMask =
            (Tensor.ones(
                  extAttentionMask.shape,
                  device: device,
                  dataType: dataType,
                ) -
                extAttentionMask) *
            minVal;
        causalMask = causalMask + extAttentionMask;
      } else if (attentionMask.shape.length == 4) {
        // If it is already 4D, we assume it is prepared
        causalMask = causalMask + attentionMask;
      }
    }

    return SimpleCausalMask(causalMask);
  }

  @override
  ({Tensor attentionOutput, Tensor attentionWeights}) perform(
    Tensor query,
    Tensor key,
    Tensor value, {
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  }) {
    context.onloadModule(this);

    final [batchSize, numHeads, qSeqLength, headDim] = query.shape;
    final [_, _, kSeqLength, _] = key.shape;

    Tensor attentionWeights = query.matmul(key.transpose(-1, -2));
    attentionWeights = attentionWeights * scaleFactor;

    if (isCausal) {
      final causalMask = premadeCausalMask
          .index([
            // Calculate for all batches
            .all,
            // Calculate for all heads
            .all,
            // Mask for query
            // During inference of decoder-only transformer, only the last token is processed (qSeqLength = 1).
            // During training of decoder-only transformer, qSeqLength = kSeqLength.
            .slice(kSeqLength - qSeqLength, kSeqLength),
            // Mask for key
            .to(kSeqLength),
          ])
          .to(device: context.device);
      // mask is 1 where we want to keep, 0 where we want to mask
      // maskedFill fills where mask is 1 (True).
      // So we want to fill where causalMask is 0.
      attentionWeights = attentionWeights.maskedFill(
        causalMask.bitwiseNot(),
        attentionWeights.dataType.fInfo.min,
      );
    }

    if (attentionMask != null) {
      attentionWeights = attentionWeights + attentionMask;
    }

    attentionWeights = attentionWeights.softmax(-1);
    attentionWeights = attnDropout.forward(attentionWeights, context: context);

    if (headMask != null) {
      attentionWeights = attentionWeights * headMask;
    }

    Tensor attentionOutput = attentionWeights.matmul(value);

    return (
      attentionOutput: attentionOutput,
      attentionWeights: attentionWeights,
    );
  }

  @override
  final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Tensor> nonTrainableParameters = [premadeCausalMask];

  @override
  final Iterable<Module> submodules = [];

  @override
  final Map<String, Tensor> meta = {};

  @override
  void resetParameters() {}
}

class GPT2AttentionMethodSDAP extends Module implements GPT2AttentionMethod {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.sdap;

  @override
  final double scaleFactor;

  @override
  final bool isCausal;

  @override
  final Dropout attnDropout;

  GPT2AttentionMethodSDAP({
    required this.scaleFactor,
    required this.isCausal,
    required this.attnDropout,
    required int maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_sdap',
  });

  @override
  /// Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that the element should take part in the attention computation, and False that it should not.
  CausalMask? makeCausalMask(
    int batchSize,
    int qLen,
    DataType dataType,
    Device device, {
    int pastKeyValuesLength = 0,
    Tensor? attentionMask,
  }) {
    // SDAP uses boolean mask
    // If fully causal and no padding mask, we can often skip the mask (return null to let SDPA use is_causal=True)
    // Checks based on transformers internal logic:
    // 1. If attention_mask is != null (padding), we usually need a mask unless we can slice it.
    //    For now, if attention_mask is present, we create a boolean mask.
    // 2. If seqLength == 1 (generation step), and fully causal, we might not need a mask if we use is_causal=True,
    //    but SDPA is_causal=True expects full causal.
    //    However, usually we pass a mask if we have padding or complex cases.

    // Logic:
    // If attentionMask is NULL and we are casual, we *might* return null to use isCausal=true in implementation.
    // However, the caller expects a mask to pass to perform.
    // In perform, we see: `attnMask: attentionMask`.
    // If we return null here, `perform` gets null.
    // `perform` uses `isCausal: isCausal`.
    // So if we return null, SDPA will use isCausal=True (if set).

    if (attentionMask == null && isCausal) {
      // Optimization: return null to use optimized kernels
      return null;
    }

    final targetLength = qLen;
    final sourceLength = qLen + pastKeyValuesLength;

    // Create boolean mask: True for keep, False for mask
    // Indices
    final rangeRow = Tensor.arange(
      0,
      targetLength,
      device: device,
      dataType: DataType.float32,
    ).view([targetLength, 1]);
    final rangeCol = Tensor.arange(
      0,
      sourceLength,
      device: device,
      dataType: DataType.float32,
    ).view([1, sourceLength]);

    // Keep condition: j <= i + pastKeyValuesLength
    Tensor mask = rangeCol
        .gt(rangeRow + pastKeyValuesLength.toDouble())
        .bitwiseNot();

    // Add padding mask
    if (attentionMask != null) {
      // attentionMask (batch, sourceDesc) -> usually 1 for keep, 0 for mask.
      // Boolean: True for keep, False for mask.
      Tensor boolAttnMask;
      if (attentionMask.dataType == DataType.boolean) {
        boolAttnMask = attentionMask;
      } else {
        boolAttnMask = attentionMask.eq(1);
      }

      if (boolAttnMask.shape.length == 2) {
        // (batch, source) -> (batch, 1, 1, source)
        boolAttnMask = boolAttnMask.view([batchSize, 1, 1, sourceLength]);
      }

      // Combine: must be kept by BOTH (AND)
      // mask is (1, 1, target, source) [implicitly broadcasted from (target, source)]
      // We view mask as (1, 1, target, source)
      mask = mask.view([1, 1, targetLength, sourceLength]);

      mask = mask.bitwiseAnd(boolAttnMask);
    } else {
      mask = mask.view([1, 1, targetLength, sourceLength]);
    }

    return SimpleCausalMask(mask);
  }

  @override
  ({Tensor attentionOutput, Tensor attentionWeights}) perform(
    Tensor query,
    Tensor key,
    Tensor value, {
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  }) {
    // SDAP usually doesn't return attention weights, we might need to handle that.
    // However, PyTorch's scaled_dot_product_attention returns only the output.
    // The interface requires attentionWeights.
    // If the interface requires weights, SDAP might not be suitable if we strictly need weights.
    // BUT usually optimization implies dropping weights.
    // We will return empty tensor for weights or throw if accessed?
    // Let's check how to get weights. If we can't, we just return empty.

    // attentionMask comes from makeCausalMask.
    // If SimpleCausalMask, verify it.
    Tensor? actualMask;
    if (attentionMask != null) {
      // Wait, the API of perform takes 'attentionMask'.
      // We should verify if the caller passes the result of makeCausalMask here.
      // If caller calls makeCausalMask, it gets a CausalMask object.
      // The perform method takes Tensor? attentionMask.
      // This implies the caller must retain the Tensor inside CausalMask and pass it here?
      // OR perform should take CausalMask?

      // As per current signature: Tensor? attentionMask.
      // So caller does:
      // mask = method.makeCausalMask(...)
      // if (mask is SimpleCausalMask) tensor = mask.mask
      // method.perform(..., attentionMask: tensor)

      actualMask = attentionMask;
    }

    Tensor attentionOutput = NNUtil.scaledDotProductAttention(
      query,
      key,
      value,
      attnMask: actualMask,
      dropoutP: attnDropout.p,
      isCausal:
          isCausal &&
          actualMask == null, // Enable isCausal ONLY if mask is null
      scale: scaleFactor,
    );

    // TODO: support headMask if SDAP supports it (it doesn't directly).
    // If headMask is provided, we might need to fallback or apply it post-hoc?
    // Applying post-hoc is too late. Pre-hoc? Modifying attnMask?
    if (headMask != null) {
      // Warning: headMask ignored in SDAP for now
      print('Warning: headMask is not supported in SDAP implementation yet.');
    }

    // SDAP does not return weights. We return a dummy tensor.
    final attentionWeights = Tensor.empty([0], device: context.device);

    return (
      attentionOutput: attentionOutput,
      attentionWeights: attentionWeights,
    );
  }

  @override
  final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Tensor> nonTrainableParameters = [];

  @override
  final Iterable<Module> submodules = [];

  @override
  final Map<String, Tensor> meta = {};

  @override
  void resetParameters() {}
}

class GPT2AttentionMethodFlashAttention2 extends GPT2AttentionMethodSDAP {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.flashAttention2;

  GPT2AttentionMethodFlashAttention2({
    required super.scaleFactor,
    required super.isCausal,
    required super.attnDropout,
    required super.maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_flash_attention_2',
  });

  @override
  CausalMask? makeCausalMask(
    int batchSize,
    int qLen,
    DataType dataType,
    Device device, {
    int pastKeyValuesLength = 0,
    Tensor? attentionMask,
  }) {
    // Flash Attention 2 usually handles causal masking internally.
    // If attentionMask is not null (padding), we might need to return it.
    // If attentionMask is null and isCausal, we return null.
    if (attentionMask == null && isCausal) {
      return null;
    }

    // Fallback to SDPA logic (parent) if we have mask
    return super.makeCausalMask(
      batchSize,
      qLen,
      dataType,
      device,
      pastKeyValuesLength: pastKeyValuesLength,
      attentionMask: attentionMask,
    );
  }
}

class GPT2AttentionFuncFlashAttention3 extends GPT2AttentionMethodSDAP {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.flashAttention3;

  GPT2AttentionFuncFlashAttention3({
    required super.scaleFactor,
    required super.isCausal,
    required super.attnDropout,
    required super.maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_flash_attention_3',
  });
}

class GPT2AttentionFuncFlexAttention extends GPT2AttentionMethodSDAP {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.flexAttention;

  GPT2AttentionFuncFlexAttention({
    required super.scaleFactor,
    required super.isCausal,
    required super.attnDropout,
    required super.maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_flex_attention',
  });

  @override
  CausalMask? makeCausalMask(
    int batchSize,
    int qLen,
    DataType dataType,
    Device device, {
    int pastKeyValuesLength = 0,
    Tensor? attentionMask,
  }) {
    // FlexAttention uses BlockMask
    // For now returning BlockCausalMask stub
    return BlockCausalMask();
  }
}

// TODO: Implement actual Paged Attention. Currently aliasing to SDAP/Eager
class GPT2AttentionMethodPagedAttention extends GPT2AttentionMethodSDAP {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.pagedAttention;

  GPT2AttentionMethodPagedAttention({
    required super.scaleFactor,
    required super.isCausal,
    required super.attnDropout,
    required super.maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_paged_attention',
  });
}

class GPT2AttentionFuncSdpaPaged extends GPT2AttentionMethodSDAP {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.sdpaPaged;

  GPT2AttentionFuncSdpaPaged({
    required super.scaleFactor,
    required super.isCausal,
    required super.attnDropout,
    required super.maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_sdpa_paged',
  });
}

class GPT2AttentionFuncEagerPaged extends GPT2AttentionMethodEagerUpscale {
  @override
  final GPT2AttentionMethodType type = GPT2AttentionMethodType.eagerPaged;

  GPT2AttentionFuncEagerPaged({
    required super.scaleFactor,
    required super.isCausal,
    required super.attnDropout,
    required super.maxPositionEmbeddings,
    super.name = 'gpt2_attention_func_eager_paged',
  });
}

abstract class CausalMask {}

class SimpleCausalMask implements CausalMask {
  final Tensor mask;

  SimpleCausalMask(this.mask);
}

class BlockCausalMask implements CausalMask {
  // TODO: implement actual block mask when FlexAttention is available
  // For now it can wrap a Tensor or be empty
  final Tensor? blockMask;

  BlockCausalMask({this.blockMask});
}
