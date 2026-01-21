import 'package:tensor/tensor.dart';
import 'dart:math' as math;

class GemmaRotaryEmbedding extends Module {
  final Tensor invFreq;

  GemmaRotaryEmbedding({required this.invFreq}) : super(name: 'rotary_emb');

  static GemmaRotaryEmbedding make({
    required int dim,
    required double base, // rope_theta
    required int maxPositionEmbeddings,
  }) {
    // Gemma uses default RoPE logic usually.
    // LlamaRotaryEmbedding uses RopeArgs which calculates invFreq.
    // We can reuse that or implement simple invFreq calculation.
    // inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))

    // Using DefaultRopeArgs logic for consistency if possible,
    // but Gemma doesn't usually use the complex scaling of Llama 3.
    // But let's stick to manual calculation to ensure Gemma correctness without dependency on Llama specific args if possible.

    // Actually, Kamma's RopeArgs might be useful if we want to support scaling later.
    // But specifically for Gemma 1, we can just compute invFreq.

    final invFreq =
        Tensor.arange(0, dim, step: 2, dataType: DataType.float32) / dim;
    // base^(-invFreq) = exp(-invFreq * log(base))
    final invFreqComputed = (invFreq * -math.log(base)).exp();

    return GemmaRotaryEmbedding(invFreq: invFreqComputed);
  }

  /// Generates the rotary positional embeddings (cos, sin) for the given position IDs.
  ({Tensor cos, Tensor sin}) forward(
    Tensor positionIds, {
    required Context context,
  }) {
    context.onloadModule(this);
    // Ensure invFreq is on device
    invFreq.to_(device: context.device);
    positionIds.to_(device: context.device);

    // positionIds is (B, S). -> (B, 1, S)
    final positionIdsExpanded = positionIds.unsqueeze(1);

    final posIdsFloat = positionIdsExpanded.to(dataType: DataType.float32);

    // invFreq is (D/2). Unsqueeze -> (1, D/2, 1).
    final invFreqExpanded = invFreq.unsqueeze(0).unsqueeze(-1);

    // freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
    final freqs = invFreqExpanded.matmul(posIdsFloat).transpose(1, 2);

    final emb = Tensor.cat([freqs, freqs], dim: -1);
    final cos = emb.cos();
    final sin = emb.sin();

    return (cos: cos, sin: sin);
  }

  static ({Tensor qEmbed, Tensor kEmbed}) applyRotaryPosEmb(
    Tensor q,
    Tensor k,
    Tensor cos,
    Tensor sin, {
    int unsqueezeDim = 1,
  }) {
    // q, k: [batch, heads, seq_len, head_dim]
    // cos, sin: [batch, seq_len, head_dim] (usually after broadcasting or repeat)

    // We need to match dimensions.
    // cos/sin from forward are [batch, 1, seq_len, head_dim]?
    // No, forward returns [batch, head_dim/2, seq_len].transpose -> [batch, seq_len, head_dim]

    // So cos/sin are [batch, seq_len, head_dim].
    // We unsqeeze at dim 1 to get [batch, 1, seq_len, head_dim] to broadcast over heads.

    final cosUnsq = cos.unsqueeze(unsqueezeDim);
    final sinUnsq = sin.unsqueeze(unsqueezeDim);

    final qEmbed = (q * cosUnsq) + (rotateHalf(q) * sinUnsq);
    final kEmbed = (k * cosUnsq) + (rotateHalf(k) * sinUnsq);

    return (qEmbed: qEmbed, kEmbed: kEmbed);
  }

  static Tensor rotateHalf(Tensor x) {
    final lastDim = x.shape.last;
    final halfDim = lastDim ~/ 2;
    final x1 = x.slice(-1, 0, end: halfDim);
    final x2 = x.slice(-1, halfDim, end: lastDim);
    final negX2 = x2 * -1.0;
    return Tensor.cat([negX2, x1], dim: -1);
  }

  @override
  final Map<String, dynamic> meta = {};

  @override
  late final Iterable<Tensor> nonTrainableParameters = [invFreq];

  @override
  final Iterable<Tensor> parameters = [];

  @override
  void resetParameters() {}

  @override
  final Iterable<Module> submodules = [];
}
