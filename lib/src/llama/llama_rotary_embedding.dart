import 'dart:math' as math;
import 'package:tensor/tensor.dart';
import 'package:kamma/src/llama/llama_config.dart';

class LlamaRotaryEmbedding {
  late final Tensor invFreq;
  late final double attentionScaling;
  final LlamaConfig config;

  LlamaRotaryEmbedding(this.config) {
    // rope_init_fn logic
    final ropeType =
        config.ropeScaling?['rope_type'] ??
        config.ropeScaling?['type'] ??
        'default';

    // Calculate invFreq based on ropeType
    if (ropeType == 'llama3') {
      _initLlama3();
    } else {
      _initDefault();
    }
  }

  void _initDefault() {
    // _compute_default_rope_parameters
    final base = config.ropeTheta;
    final headDim = config.headDim;
    final dim = headDim; // Assuming partial_rotary_factor is 1.0 mostly

    // inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    // Tensor.arange(start, end, {step, ...})
    final indices = Tensor.arange(0, dim, step: 2, datatype: DataType.float32);
    final exponent = indices / dim;

    // base ** exponent
    // We can use Tensor.pow(exponent) called on base tensor, or base.pow(exponent)
    // Since base is scalar, let's create a tensor from it? Or just use math if possible?
    // Exponent is a Tensor.
    // Let's do: base ^ exponent.
    // Tensor.pow usually supports scalar exponent. What about tensor exponent?
    // If not, we can do exp(exponent * ln(base)).
    // invFreq = 1.0 / (base ** exponent)

    // Tensor.pow(dynamic exponent). If exponent is Tensor, it might work?
    // Let's assume indices.pow is not what we want (that's indices^exponent).
    // We want base^indices.
    // base^(x) = exp(x * ln(base))
    final lnBase = math.log(base);
    final denom = (exponent * lnBase).exp();

    invFreq = Tensor.full([1], 1.0) / denom;
    // Or just 1.0 / denom if division by tensor works for scalar (it usually does)
    attentionScaling = 1.0;
  }

  void _initLlama3() {
    // _compute_llama3_parameters
    // First get default invFreq
    _initDefault();
    final defaultInvFreq = invFreq;

    final factor = (config.ropeScaling!['factor'] as num).toDouble();
    final lowFreqFactor = (config.ropeScaling!['low_freq_factor'] as num)
        .toDouble();
    final highFreqFactor = (config.ropeScaling!['high_freq_factor'] as num)
        .toDouble();
    final oldContextLen =
        (config.ropeScaling!['original_max_position_embeddings'] as num)
            .toDouble();

    final lowFreqWavelen = oldContextLen / lowFreqFactor;
    final highFreqWavelen = oldContextLen / highFreqFactor;

    final wavelen = defaultInvFreq.pow(-1.0) * (2 * math.pi);

    // inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    final wavelenGtLow = wavelen.gt(lowFreqWavelen);
    // Condition.where(true_val, false_val)
    var invFreqLlama = wavelenGtLow.where(
      defaultInvFreq / factor,
      defaultInvFreq,
    );

    // smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    final smoothFactor =
        (wavelen.pow(-1.0) * oldContextLen - lowFreqFactor) /
        (highFreqFactor - lowFreqFactor);

    // smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    final smoothedInvFreq =
        (smoothFactor * -1.0 + 1.0) * (invFreqLlama / factor) +
        (smoothFactor * invFreqLlama);

    // is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    // which is: wavelen >= high_freq_wavelen AND wavelen <= low_freq_wavelen
    // wavelen >= high => !(wavelen < high).
    final wavelenGeHigh = wavelen.lt(highFreqWavelen).bitwiseNot();
    // wavelen <= low => !(wavelen > low)
    final wavelenLeLow = wavelen.gt(lowFreqWavelen).bitwiseNot();

    final isMediumFreq = wavelenGeHigh.bitwiseAnd(wavelenLeLow);

    // inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
    invFreq = isMediumFreq.where(smoothedInvFreq, invFreqLlama);
    attentionScaling =
        1.0; // Llama3 doesn't return attention_factor other than 1.0 in the code I read?
    // checking logic: return inv_freq_llama, attention_factor (which was 1.0 from default)
  }

  // Forward / call is typically embedding, but here is helper to get cos/sin
  (Tensor, Tensor) call(Tensor x, Tensor positionIds) {
    // inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    // position_ids_expanded = position_ids[:, None, :].float()

    // Simplified logic for Dart Tensor:
    // positionIds is (B, S). -> (B, 1, S)
    final positionIdsExpanded = positionIds.unsqueeze(1);

    // freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
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

// Helper functions (could be static or top level)

Tensor rotateHalf(Tensor x) {
  // x1 = x[..., : x.shape[-1] // 2]
  // x2 = x[..., x.shape[-1] // 2 :]
  // return torch.cat((-x2, x1), dim=-1)
  final lastDim = x.shape.last;
  final halfDim = lastDim ~/ 2;
  // x1 = x[..., : x.shape[-1] // 2]
  // x2 = x[..., x.shape[-1] // 2 :]
  // Tensor slice(int dim, int start, {int step = 1, int? end})
  final x1 = x.slice(-1, 0, end: halfDim);
  final x2 = x.slice(-1, halfDim, end: lastDim);
  // -x2 -> x2 * -1.0
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
  // cos = cos.unsqueeze(unsqueeze_dim)
  // sin = sin.unsqueeze(unsqueeze_dim)

  final cosUnsq = cos.unsqueeze(unsqueezeDim);
  final sinUnsq = sin.unsqueeze(unsqueezeDim);

  // q_embed = (q * cos) + (rotate_half(q) * sin)
  final qEmbed = (q * cosUnsq) + (rotateHalf(q) * sinUnsq);
  final kEmbed = (k * cosUnsq) + (rotateHalf(k) * sinUnsq);

  return (qEmbed, kEmbed);
}
