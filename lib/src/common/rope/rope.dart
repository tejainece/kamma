import 'dart:math' as math;
import 'package:kamma/kamma.dart';

abstract class RopeArgs {
  static RopeArgs parse(Map<String, dynamic> map) {
    final ropeType = map['rope_type'] ?? map['type'] ?? 'default';
    if (ropeType == 'default') {
      return DefaultRopeArgs.instance;
    } else if (ropeType == 'llama3') {
      return Llama3RopeArgs.fromMap(map);
    }
    throw ArgumentError('Unknown rope type: $ropeType');
  }
}

class DefaultRopeArgs implements RopeArgs {
  const DefaultRopeArgs._();

  static const instance = DefaultRopeArgs._();

  String get name => 'default';

  // _compute_default_rope_parameters
  ({Tensor invFreq, double attentionScaling}) compute(int dim, double base) {
    final indices = Tensor.arange(0, dim, step: 2, dataType: DataType.float32);
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

    final invFreq = Tensor.full([1], 1.0) / denom;
    // Or just 1.0 / denom if division by tensor works for scalar (it usually does)
    final attentionScaling = 1.0;

    return (invFreq: invFreq, attentionScaling: attentionScaling);
  }
}

class Llama3RopeArgs implements RopeArgs {
  final double factor;
  final double lowFreqFactor;
  final double highFreqFactor;
  final double originalMaxPositionEmbeddings;

  Llama3RopeArgs({
    required this.factor,
    required this.lowFreqFactor,
    required this.highFreqFactor,
    required this.originalMaxPositionEmbeddings,
  });

  String get name => 'llama3';

  /// _compute_llama3_parameters
  ({Tensor invFreq, double attentionScaling}) compute(
    int dim,
    double base,
    int maxPositionEmbeddings,
  ) {
    var (:invFreq, :attentionScaling) = DefaultRopeArgs.instance.compute(
      dim,
      base,
    );

    // DefaultRopeArgs returns attentionScaling=1.0.
    // We can keep using it or override if needed.

    final defaultInvFreq = invFreq;

    final lowFreqWavelen = originalMaxPositionEmbeddings / lowFreqFactor;
    final highFreqWavelen = originalMaxPositionEmbeddings / highFreqFactor;

    final wavelen = defaultInvFreq.pow(-1.0) * (2 * math.pi);

    // if wavelen > lowFreqWavelen, then invFreqLlama = defaultInvFreq / ropeScaling.factor, else invFreqLlama = defaultInvFreq
    final wavelenGtLow = wavelen.gt(lowFreqWavelen);
    var invFreqLlama = wavelenGtLow.where(
      defaultInvFreq / factor,
      defaultInvFreq,
    );

    final smoothFactor =
        (wavelen.pow(-1.0) * originalMaxPositionEmbeddings - lowFreqFactor) /
        (highFreqFactor - lowFreqFactor);

    final smoothedInvFreq =
        (smoothFactor * -1.0 + 1.0) * (invFreqLlama / factor) +
        (smoothFactor * invFreqLlama);

    // Isolate the 'medium' frequency band where we need to blend the original and scaled frequencies.
    // This band includes wavelengths that are too long to be considered 'high frequency' (unscaled)
    // but too short to be considered 'low frequency' (fully scaled).
    final wavelenGeHigh = wavelen.lt(highFreqWavelen).bitwiseNot();
    final wavelenLeLow = wavelen.gt(lowFreqWavelen).bitwiseNot();
    final isMediumFreq = wavelenGeHigh.bitwiseAnd(wavelenLeLow);

    invFreq = isMediumFreq.where(
      smoothedInvFreq, // Explicitly enforce appropriate scaling
      invFreqLlama,
    );

    // Debug prints
    print('Llama3RopeArgs.compute:');
    print('  dim: $dim, base: $base, maxPos: $maxPositionEmbeddings');
    print(
      '  lowFreqFactor: $lowFreqFactor, highFreqFactor: $highFreqFactor, factor: $factor',
    );
    print('  originalMaxPos: $originalMaxPositionEmbeddings');

    final changed = (invFreq - defaultInvFreq).abs().max().scalar as double;
    print('  invFreq diff from default: $changed');

    return (invFreq: invFreq, attentionScaling: 1.0);
  }

  static Llama3RopeArgs fromMap(Map<String, dynamic> map) {
    return Llama3RopeArgs(
      factor: (map['factor'] as num).toDouble(),
      lowFreqFactor: (map['low_freq_factor'] as num).toDouble(),
      highFreqFactor: (map['high_freq_factor'] as num).toDouble(),
      originalMaxPositionEmbeddings:
          (map['original_max_position_embeddings'] as num).toDouble(),
    );
  }
}
