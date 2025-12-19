import 'dart:math' as math;
import 'package:tensor/tensor.dart';
import 'deepseek_config.dart';

class DeepSeekRotaryEmbedding {
  late final Tensor invFreq;
  late final double attentionScaling;
  final DeepSeekConfig config;

  DeepSeekRotaryEmbedding(this.config) {
    _initDefault();
  }

  void _initDefault() {
    final base = config.rotaryEmbBase;
    final dim = config.ropeHeadDim; // RoPE is applied only to this dimension

    // inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    final indices = Tensor.arange(0, dim, step: 2, datatype: DataType.float32);
    final exponent = indices / dim;

    final lnBase = math.log(base);
    final denom = (exponent * lnBase).exp();

    invFreq = Tensor.full([1], 1.0) / denom;
    attentionScaling = 1.0;
  }

  // Forward / call is typically embedding, but here is helper to get cos/sin
  (Tensor, Tensor) call(Tensor x, Tensor positionIds) {
    // positionIds is (B, S). -> (B, 1, S)
    final positionIdsExpanded = positionIds.unsqueeze(1);

    // Tensor operations usually promote, but better be safe.
    final posIdsFloat = positionIdsExpanded.to(dataType: DataType.float32);

    // invFreq is (D/2). Unsqueeze -> (1, D/2, 1).
    final invFreqExpanded = invFreq.unsqueeze(0).unsqueeze(-1);

    // (1, D/2, 1) @ (B, 1, S) -> Broadcasting invFreq to (B, D/2, 1)
    // Matmul: (B, D/2, 1) * (B, 1, S) -> (B, D/2, S)
    final freqs = invFreqExpanded.matmul(posIdsFloat).transpose(1, 2);

    // emb = torch.cat((freqs, freqs), dim=-1)
    final emb = Tensor.cat([freqs, freqs], dim: -1);

    // cos = emb.cos() * self.attention_scaling
    // sin = emb.sin() * self.attention_scaling
    final cos = emb.cos() * attentionScaling;
    final sin = emb.sin() * attentionScaling;

    return (cos, sin); // Standard output, user can cast if needed
  }
}

// Helper functions
Tensor rotateHalf(Tensor x) {
  final lastDim = x.shape.last;
  final halfDim = lastDim ~/ 2;
  final x1 = x.slice(-1, 0, end: halfDim);
  final x2 = x.slice(-1, halfDim, end: lastDim);
  final negX2 = x2 * -1.0;
  return Tensor.cat([negX2, x1], dim: -1);
}

(Tensor, Tensor) applyRotaryPosEmb(
  Tensor q,
  Tensor k,
  Tensor cos,
  Tensor sin, {
  int unsqueezeDim = 1,
}) {
  final cosUnsq = cos.unsqueeze(unsqueezeDim);
  final sinUnsq = sin.unsqueeze(unsqueezeDim);

  final qEmbed = (q * cosUnsq) + (rotateHalf(q) * sinUnsq);
  final kEmbed = (k * cosUnsq) + (rotateHalf(k) * sinUnsq);

  return (qEmbed, kEmbed);
}
