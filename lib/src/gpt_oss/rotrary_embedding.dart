import 'dart:math';

import 'package:tensor/tensor.dart';

class GptOssRotaryEmbedding extends Module {
  final double base;
  final int dim;
  final int maxPositionEmbeddings;
  final RoPEInvFreq ropeInvFreq;
  late final Tensor invFreq;

  GptOssRotaryEmbedding({
    required super.name,
    required this.dim,
    this.maxPositionEmbeddings = 2048,
    this.base = 10000.0,
    this.ropeInvFreq = RoPEInvFreqDefault.instance,
  }) {
    _resetParameters();
  }

  void _resetParameters() {
    final result = ropeInvFreq.perform(theta: base, dim: dim);
    invFreq = result.invFreq;
    // TODO remove if possible
    _updateCosSinCache(maxPositionEmbeddings);
  }

  void _updateCosSinCache(int seqLen) {
    // t = arange(seqLen)
    final tList = List.generate(seqLen, (i) => i.toDouble());
    // Create t on correct device to avoid mismatch with invFreq
    final t = Tensor.from(tList, [
      seqLen,
    ], datatype: DataType.float32).to(device: invFreq.device);

    // freqs = outer(t, inv_freq)
    // t: [seq], invFreq: [dim/2]
    // outer: [seq, dim/2] -> t.unsqueeze(1) * invFreq.unsqueeze(0)
    final freqs = t.unsqueeze(1) * invFreq.unsqueeze(0);

    // Different from RoPE usually?
    // Transformers: emb = cat((freqs, freqs), dim=-1)
    final emb = Tensor.cat([freqs, freqs], dim: -1); // [seq, dim]

    // We keep them on CPU usually until needed, or move to device if module is moved.
  }

  @override
  void to_(Device device, {bool cascade = false}) {
    super.to_(device, cascade: cascade);
    invFreq = invFreq.to(device: device);
  }

  // Helper to rotate half
  static Tensor rotateHalf(Tensor x) {
    // x: [..., dim]
    // split into x1, x2
    final chunks = x.chunk(2, dim: -1);
    final x1 = chunks[0];
    final x2 = chunks[1];

    // returns [-x2, x1]
    // -x2
    final negX2 =
        Tensor.zeros(x2.shape, device: x2.device, datatype: x2.dataType) - x2;
    return Tensor.cat([negX2, x1], dim: -1);
  }

  ({Tensor cos, Tensor sin}) forward(
    Tensor positionIds, {
    required int seqLen,
  }) {
    // Reference implementation:
    // inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    // position_ids_expanded = position_ids[:, None, :].float()
    // freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
    // emb = freqs
    // cos = emb.cos() * self.attention_scaling
    // sin = emb.sin() * self.attention_scaling

    // invFreq: [dim/2]
    // positionIds: [batch, seq]

    // Create views for matmul
    // invFreq: [1, dim/2, 1]
    final invFreqView = invFreq
        .view([1, invFreq.shape[0], 1])
        .expand([positionIds.shape[0], -1, 1])
        .to(device: positionIds.device); // Ensure on same device as input

    // positionIds: [batch, 1, seq]
    final batchSize = positionIds.shape[0];
    final posView = positionIds
        .view([batchSize, 1, seqLen])
        .to(dataType: DataType.float32);

    // Matmul with broadcasting
    // [1, dim/2, 1] @ [batch, 1, seq] -> [batch, dim/2, seq]
    final freqs = invFreqView.matmul(posView);

    // Transpose to [batch, seq, dim/2]
    final emb = freqs.transpose(1, 2);

    // Compute cos/sin
    final cos = emb.cos();
    final sin = emb.sin();

    return (cos: cos, sin: sin);
  }

  // inplace apply
  static void applyRotaryPosEmb(Tensor q, Tensor k, Tensor cos, Tensor sin) {
    // q, k: [batch, head, seq, head_dim]
    // cos, sin: [batch, seq, head_dim/2]

    // Reshape cos, sin for broadcasting over heads: [batch, 1, seq, head_dim/2]
    final cosView = cos.view([cos.shape[0], 1, cos.shape[1], cos.shape[2]]);
    final sinView = sin.view([sin.shape[0], 1, sin.shape[1], sin.shape[2]]);

    Tensor rotate(Tensor x) {
      final chunks = x.chunk(2, dim: -1);
      final x1 = chunks[0];
      final x2 = chunks[1];

      final out1 = x1 * cosView - x2 * sinView;
      final out2 = x2 * cosView + x1 * sinView;
      return Tensor.cat([out1, out2], dim: -1);
    }

    final qEmbed = rotate(q);
    final kEmbed = rotate(k);

    q.copy_(qEmbed);
    k.copy_(kEmbed);
  }

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  late final Iterable<Tensor> nonTrainableParameters = [invFreq];

  @override
  late final Iterable<Module> submodules = const [];

  @override
  void resetParameters() {
    _resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {
    'dim': dim,
    'maxPositionEmbeddings': maxPositionEmbeddings,
    'base': base,
    'rope_inv_freq': ropeInvFreq.name,
  };
}

abstract class RoPEInvFreq {
  String get name;

  ({Tensor invFreq, double attentionFactor}) perform({
    required double theta,
    required int dim,
  });

  const RoPEInvFreq();
}

/// Computes the inverse frequencies according to the original RoPE implementation
class RoPEInvFreqDefault implements RoPEInvFreq {
  @override
  final String name = 'default';

  @override
  ({Tensor invFreq, double attentionFactor}) perform({
    required double theta,
    required int dim,
  }) {
    // inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
    // Only even indices: 0, 2, ..., dim-2
    // Computation: exp( (arange(0, dim, 2) / dim) * -log(theta) )

    // 1. arange(0, dim, 2)
    final indices = Tensor.arange(0, dim, step: 2, datatype: DataType.float32);

    // 2. indices / dim
    final ratio = indices / dim.toDouble();

    // 3. -log(theta)
    final negLogTheta = -log(theta);

    // 4. exp(ratio * negLogTheta) == theta ** (-ratio) == 1 / theta ** ratio
    final invFreq = (ratio * negLogTheta).exp();

    return (invFreq: invFreq, attentionFactor: 1.0);
  }

  const RoPEInvFreqDefault._();
  static const instance = RoPEInvFreqDefault._();
}
