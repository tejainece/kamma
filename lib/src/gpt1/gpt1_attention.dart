import 'dart:math';
import 'package:kamma/kamma.dart';

class OpenAIGPTAttention extends Module {
  final int nHead;
  final int nCtx;
  final bool scale;
  final LinearLayer cAttn;
  final LinearLayer cProj;
  final Dropout attnDropout;
  final Dropout residDropout;

  OpenAIGPTAttention({
    required super.name,
    required this.cAttn,
    required this.cProj,
    required this.attnDropout,
    required this.residDropout,
    required this.nHead,
    required this.nCtx,
    this.scale = true,
  }) {
    // if (cAttn.inFeatures % nHead != 0) { ... }
  }

  int get embedDim => cAttn.inFeatures;
  int get headDim => embedDim ~/ nHead;

  Tensor _attn(
    Tensor q,
    Tensor k,
    Tensor v, {
    Tensor? attentionMask,
    Tensor? headMask,
    required Context context,
  }) {
    // q: [batch, nHead, seqLen, headDim]
    // k: [batch, nHead, seqLen, headDim]
    // w: [batch, nHead, seqLen, seqLen]
    Tensor w = q.matmul(k.transpose(-1, -2));

    if (scale) {
      w = w / sqrt(headDim.toDouble());
    }

    // Causal mask implementation
    // Transformers uses a buffer `bias` which is a lower triangular matrix of ones.
    // w = w * bias + -1e9 * (1 - bias)
    // We create the mask on the fly or we could have stored it.
    // Since nCtx is fixed, we can create it.
    // However, recreating it every time might be slow? Tensor.ones is cheap enough.

    // Create causal mask
    // We need [seqLen, seqLen] lower triangular
    final seqLen = q.shape[2];
    // mask where row >= col
    // Use newly implemented tril
    Tensor b = Tensor.ones(
      [seqLen, seqLen],
      dataType: w.dataType,
      device: w.device,
    );
    b = b.tril();

    // Explicit reshape to [1, 1, seqLen, seqLen] for broadcasting
    b = b.reshape([1, 1, seqLen, seqLen]);

    // Apply causal mask logic: w * b + -1e9 * (1 - b)
    // Actually, w * b sets non-causal to 0. Then (1-b) selects non-causal positions and sets them to -1e9.
    // Result: causal kept as is, non-causal becomes -1e9.
    w = w * b + (b * -1.0 + 1.0) * -1e9;

    // Apply attention mask (padding mask)
    if (attentionMask != null) {
      // attentionMask assumed to be [batch, 1, 1, seqLen] or similar broadcastable
      // In HF: w = w + attention_mask
      w = w + attentionMask;
    }

    w = w.softmax(-1);
    w = attnDropout.forward(w, context: context);

    if (headMask != null) {
      w = w * headMask;
    }

    Tensor output = w.matmul(v);
    return output;
  }

  // Helper to split heads: [batch, seqLen, feature] -> [batch, nHead, seqLen, feature/nHead]
  Tensor _splitHeads(Tensor x, int knHead) {
    final shape = x.shape;
    final batchSize = shape[0];
    final seqLen = shape[1];
    final feature = shape[2];
    final kHeadDim = feature ~/ knHead;

    // new shape: [batch, seqLen, nHead, headDim]
    x = x.view([batchSize, seqLen, knHead, kHeadDim]);
    // permute to [batch, nHead, seqLen, headDim]
    return x.permute([0, 2, 1, 3]);
  }

  // Helper to merge heads: [batch, nHead, seqLen, headDim] -> [batch, seqLen, feature]
  Tensor _mergeHeads(Tensor x) {
    // x: [batch, nHead, seqLen, headDim]
    // permute to [batch, seqLen, nHead, headDim]
    x = x.permute([0, 2, 1, 3]).contiguous();
    final shape = x.shape;
    final batchSize = shape[0];
    final seqLen = shape[1];
    // reshape: [batch, seqLen, nHead * headDim]
    return x.view([batchSize, seqLen, shape[2] * shape[3]]);
  }

  Tensor forward(
    Tensor hiddenStates, {
    Tensor? attentionMask,
    Tensor? headMask,
    bool outputAttentions = false,
    required Context context,
  }) {
    context.onloadModule(this);

    // cAttn projects to 3 * n_embd
    Tensor qkv = cAttn.forward(hiddenStates, context: context);
    // split into q, k, v
    List<Tensor> parts = qkv.splitEqually(embedDim, dim: 2);
    Tensor q = parts[0];
    Tensor k = parts[1];
    Tensor v = parts[2];

    q = _splitHeads(q, nHead);
    k = _splitHeads(k, nHead);
    v = _splitHeads(v, nHead);

    Tensor attnOutput = _attn(
      q,
      k,
      v,
      attentionMask: attentionMask,
      headMask: headMask,
      context: context,
    );

    attnOutput = _mergeHeads(attnOutput);
    attnOutput = cProj.forward(attnOutput, context: context);
    attnOutput = residDropout.forward(attnOutput, context: context);

    return attnOutput;
  }

  @override
  void resetParameters() {
    cAttn.resetParameters();
    cProj.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {
    "nHead": nHead,
    "nCtx": nCtx,
    "scale": scale,
    "embedDim": cAttn.inFeatures,
  };

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [
    cAttn,
    cProj,
    attnDropout,
    residDropout,
  ];

  static OpenAIGPTAttention make({
    required String name,
    required int nEmbd,
    required int nHead,
    required int nCtx,
    required double attnPdrop,
    required double residPdrop,
    bool scale = true,
  }) {
    final cAttn = LinearLayer.make(
      name: 'c_attn',
      inFeatures: nEmbd,
      outFeatures: 3 * nEmbd,
    );
    final cProj = LinearLayer.make(
      name: 'c_proj',
      inFeatures: nEmbd,
      outFeatures: nEmbd,
    );

    return OpenAIGPTAttention(
      name: name,
      cAttn: cAttn,
      cProj: cProj,
      attnDropout: Dropout(attnPdrop),
      residDropout: Dropout(residPdrop),
      nHead: nHead,
      nCtx: nCtx,
      scale: scale,
    );
  }

  static Future<OpenAIGPTAttention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int nEmbd,
    required int nHead,
    required int nCtx,
    required double attnPdrop,
    required double residPdrop,
    bool scale = true,
  }) async {
    final cAttn = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix:
          '$prefix'
          'c_attn.',
      name: 'c_attn',
    );
    final cProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix:
          '$prefix'
          'c_proj.',
      name: 'c_proj',
    );

    return OpenAIGPTAttention(
      name: name,
      cAttn: cAttn,
      cProj: cProj,
      attnDropout: Dropout(attnPdrop),
      residDropout: Dropout(residPdrop),
      nHead: nHead,
      nCtx: nCtx,
      scale: scale,
    );
  }
}
