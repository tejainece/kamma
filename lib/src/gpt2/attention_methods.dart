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
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
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
  ({Tensor attentionOutput, Tensor attentionWeights}) perform(
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
    if (!isCausal) {
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
      attentionWeights = causalMask.where(
        attentionWeights,
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
  ({Tensor attentionOutput, Tensor attentionWeights}) perform(
    Tensor query,
    Tensor key,
    Tensor value, {
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  }) {
    final [batchSize, numHeads, qSeqLength, headDim] = query.shape;
    final [_, _, kSeqLength, _] = key.shape;

    Tensor attentionWeights = query.matmul(key.transpose(-1, -2));
    attentionWeights = attentionWeights * scaleFactor;

    if (!isCausal) {
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
      attentionWeights = causalMask.where(
        attentionWeights,
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

    Tensor attentionOutput = NNUtil.scaledDotProductAttention(
      query,
      key,
      value,
      attnMask: attentionMask,
      dropoutP: attnDropout.p,
      isCausal: isCausal,
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
