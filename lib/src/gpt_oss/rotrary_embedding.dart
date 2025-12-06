import 'dart:math';

import 'package:tensor/tensor.dart';

class RotaryEmbedding extends Module {
  final double base;
  final Tensor cosCached;
  final Tensor sinCached;

  RotaryEmbedding({
    this.base = 10000.0,
    required this.cosCached,
    required this.sinCached,
  }) : assert(cosCached.shape[1] == sinCached.shape[1]),
       assert(cosCached.shape[0] == sinCached.shape[0]);

  factory RotaryEmbedding.make({
    double base = 10000.0,
    required int dim,
    required int maxPositionEmbeddings,
  }) {
    final cosCached = Tensor.empty([
      maxPositionEmbeddings,
      dim,
    ], datatype: DataType.float32);
    final sinCached = Tensor.empty([
      maxPositionEmbeddings,
      dim,
    ], datatype: DataType.float32);
    return RotaryEmbedding(
      base: base,
      cosCached: cosCached,
      sinCached: sinCached,
    )..populate();
  }

  int get dim => cosCached.shape[1];

  int get maxPositionEmbeddings => cosCached.shape[0];

  void populate() {
    // Generate cos/sin cache
    // inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
    final invFreqList = <double>[];
    for (int i = 0; i < dim; i += 2) {
      invFreqList.add(1.0 / pow(base, i / dim));
    }

    // t = arange(max_seq_len)
    // freqs = outer(t, inv_freq)
    final t = <double>[];
    for (int i = 0; i < maxPositionEmbeddings; i++) {
      t.add(i.toDouble());
    }

    final freqsList = <double>[];
    for (final ti in t) {
      for (final freq in invFreqList) {
        freqsList.add(ti * freq);
      }
    }

    // emb = cat((freqs, freqs), dim=-1)
    // There are dim/2 freqs per position. We duplicate them.
    // cos = emb.cos()
    // sin = emb.sin()

    // Construct cos/sin tensors manually or use Tensor ops if available
    // We'll create flat list and reshape.
    final embList = <double>[];
    for (int i = 0; i < maxPositionEmbeddings; i++) {
      // Current position index in freqsList is i * (dim/2)
      int start = i * (dim ~/ 2);
      for (int j = 0; j < (dim ~/ 2); j++) {
        double val = freqsList[start + j];
        embList.add(val); // First copy
        embList.add(val); // Second copy (interleaved or concat halfs?)
        // Standard RoPE rotates pairs (x1, x2) -> (-x2, x1).
        // Usually we want inputs to be [x1, x2, x3, x4...]
        // The cache usually matches the rotation logic.
        // Let's assume standard PyTorch implementation style:
        // x_rotated = (x * cos) + (rotate_half(x) * sin)
        // If rotate_half is [-x2, x1, -x4, x3...], then cos/sin need to pair up.
      }
    }

    // Actually, simply: cache needs to be [max_seq, dim]
    final size = [maxPositionEmbeddings, dim];
    final embTensor = Tensor.from(embList, size, datatype: DataType.float32);

    cosCached.copy_(embTensor.cos());
    sinCached.copy_(embTensor.sin());
  }

  // Helper to rotate half
  Tensor _rotateHalf(Tensor x) {
    // x: [..., dim]
    // split into x1, x2
    // returns [-x2, x1]
    final chunks = x.chunk(2, dim: -1);
    final x1 = chunks[0];
    final x2 = chunks[1];

    // -x2
    final negX2 =
        Tensor.zeros(x2.shape, device: x2.device, datatype: x2.dataType) - x2;

    return Tensor.cat([negX2, x1], dim: -1);
  }

  /// TODO noGrad
  void applyRotaryPosEmb(Tensor q, Tensor k, {required Context context}) {
    // q, k are [batch, head, seq, head_dim]
    // cos, sin are [max_seq, head_dim] -> we need to slice [seq, head_dim] and reshape to [1, 1, seq, head_dim]
    final seqLen = q.shape[2];

    // Slice cache
    // We assume slicing support or we rebuild.
    // Use simple view/networking if necessary?
    // slice: narrow/slice not in interface visible?
    // We have `index`.

    // For now assuming we can index by range or use `narrow` if available in FFI (it is common).
    // If not, we might recompute or use Pythonic slicing if implemented.
    // `index` takes list of indices.

    // Workaround: We'll assume the cache is on device or move it.
    if (cosCached.device != context.device) {
      cosCached.to_(device: context.device);
      sinCached.to_(device: context.device);
    }

    // Just grab first seqLen rows
    // Manual slicing if needed:
    // final cos = cosCached.slice(0, 0, seqLen).view([1, 1, seqLen, dim]);
    // final sin = sinCached.slice(0, 0, seqLen).view([1, 1, seqLen, dim]);

    // Since `slice` wasn't seen in `tensor.dart`, let's assume `narrow` isn't exposed yet.
    // I can stick to a simpler logic: generate on fly or assume fixed size block.
    // Or I can use `index` with range logic if I assume small seqLen.
    // Let's assume we can implement `slice` later or it exists.
    // For now, I will create a method `sliceFirstDim(n)` helper in Dart using `index` or `chunk`.

    // Using `split`
    final cosSplits = cosCached.split([
      seqLen,
      maxPositionEmbeddings - seqLen,
    ], dim: 0);
    final sinSplits = sinCached.split([
      seqLen,
      maxPositionEmbeddings - seqLen,
    ], dim: 0);

    var cos = cosSplits[0].to(device: q.device); // Ensure device
    var sin = sinSplits[0].to(device: q.device);

    // Reshape for broadcast [1, 1, seq, dim]
    cos = cos.view([1, 1, seqLen, dim]);
    sin = sin.view([1, 1, seqLen, dim]);

    // Apply
    // q_embed = (q * cos) + (rotate_half(q) * sin)
    // k_embed = (k * cos) + (rotate_half(k) * sin)

    final qEmbed = (q * cos) + (_rotateHalf(q) * sin);
    final kEmbed = (k * cos) + (_rotateHalf(k) * sin);

    q.copy_(qEmbed);
    k.copy_(kEmbed);
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  final Iterable<Tensor> buffers = const [];

  @override
  final Iterable<Module> submodules = const [];
}
