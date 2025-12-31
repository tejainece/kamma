import 'dart:math';
import 'package:tensor/tensor.dart';

/// 3D Rotary Position Embedding (RoPE) for S3-DiT
///
/// Handles three dimensions:
/// - Spatial X (for image tokens)
/// - Spatial Y (for image tokens)
/// - Temporal/Sequence (for text and sequential tokens)
///
/// This allows unified modeling of multimodal sequences within standard transformer blocks.
class RoPE3D {
  final int dim;
  final double theta;
  final int maxSeqLen;

  late final Tensor _freqsX;
  late final Tensor _freqsY;
  late final Tensor _freqsT;

  RoPE3D({required this.dim, this.theta = 10000.0, this.maxSeqLen = 8192}) {
    // Compute frequency bands for each dimension
    // Each dimension gets dim/6 frequency bands (since we have 3 dimensions and need cos/sin pairs)
    final int dimPerAxis = dim ~/ 6;

    _freqsX = _computeFreqs(dimPerAxis);
    _freqsY = _computeFreqs(dimPerAxis);
    _freqsT = _computeFreqs(dimPerAxis);
  }

  /// Compute frequency bands for RoPE
  Tensor _computeFreqs(int dimSize) {
    // freq = 1.0 / (theta^(2i/dim)) for i in [0, dimSize/2)
    final List<double> freqs = [];
    for (int i = 0; i < dimSize ~/ 2; i++) {
      final double freq = 1.0 / pow(theta, (2.0 * i) / dimSize);
      freqs.add(freq);
    }
    return Tensor.from(freqs, [freqs.length], dataType: DataType.float32);
  }

  /// Apply 3D rotary position embedding to query or key tensors
  ///
  /// Args:
  ///   x: Input tensor of shape [batch, seqLen, numHeads, headDim]
  ///   posX: X position indices [batch, seqLen] or [seqLen]
  ///   posY: Y position indices [batch, seqLen] or [seqLen]
  ///   posT: Temporal/sequence position indices [batch, seqLen] or [seqLen]
  ///
  /// Returns:
  ///   Tensor with rotary embeddings applied
  Tensor apply(
    Tensor x, {
    required Tensor posX,
    required Tensor posY,
    required Tensor posT,
  }) {
    // Split head dimension into 3 parts for x, y, t
    final int headDim = x.shape.last;
    final int dimPerAxis = headDim ~/ 3;

    // Split x into three parts along head dimension
    final xParts = x.split([dimPerAxis, dimPerAxis, dimPerAxis], dim: -1);
    final Tensor xX = xParts[0];
    final Tensor xY = xParts[1];
    final Tensor xT = xParts[2];

    // Apply rotary embedding to each part
    final Tensor rotatedX = _applyRotary(xX, posX, _freqsX);
    final Tensor rotatedY = _applyRotary(xY, posY, _freqsY);
    final Tensor rotatedT = _applyRotary(xT, posT, _freqsT);

    // Concatenate back together
    return Tensor.cat([rotatedX, rotatedY, rotatedT], dim: -1);
  }

  /// Apply rotary embedding to a single dimension
  Tensor _applyRotary(Tensor x, Tensor positions, Tensor freqs) {
    // x shape: [batch, seqLen, numHeads, dimPerAxis]
    // positions shape: [batch, seqLen] or [seqLen]

    // Compute angles: positions * freqs
    // freqs shape: [dimPerAxis/2]
    // We need to broadcast positions to match

    // Expand positions to [batch, seqLen, 1]
    Tensor pos = positions;
    if (pos.shape.length == 1) {
      pos = pos.unsqueeze(0); // [1, seqLen]
    }
    pos = pos.unsqueeze(-1); // [batch, seqLen, 1]

    // Compute angles [batch, seqLen, dimPerAxis/2]
    final Tensor angles =
        pos.to(dataType: DataType.float32) * freqs.unsqueeze(0).unsqueeze(0);

    // Compute cos and sin
    final Tensor cosAngles = angles.cos();
    final Tensor sinAngles = angles.sin();

    // Split x into even and odd indices
    // x shape: [batch, seqLen, numHeads, dimPerAxis]
    final int dimPerAxis = x.shape.last;
    final xReshaped = x.view([
      x.shape[0],
      x.shape[1],
      x.shape[2],
      dimPerAxis ~/ 2,
      2,
    ]);

    final Tensor x0 = xReshaped.select(-1, 0); // Even indices
    final Tensor x1 = xReshaped.select(-1, 1); // Odd indices

    // Expand cos and sin to match x dimensions
    // cosAngles, sinAngles: [batch, seqLen, dimPerAxis/2]
    // Need: [batch, seqLen, numHeads, dimPerAxis/2]
    final Tensor cosExpanded = cosAngles.unsqueeze(2);
    final Tensor sinExpanded = sinAngles.unsqueeze(2);

    // Apply rotation
    // rotated_even = x0 * cos - x1 * sin
    // rotated_odd = x0 * sin + x1 * cos
    final Tensor rotatedEven = x0 * cosExpanded - x1 * sinExpanded;
    final Tensor rotatedOdd = x0 * sinExpanded + x1 * cosExpanded;

    // Stack back together
    final Tensor rotated = Tensor.stack([rotatedEven, rotatedOdd], dim: -1);

    // Reshape back to original shape
    return rotated.view(x.shape);
  }

  /// Create position indices for a 2D image grid
  ///
  /// Args:
  ///   height: Image height in tokens
  ///   width: Image width in tokens
  ///   batchSize: Batch size
  ///
  /// Returns:
  ///   Tuple of (posX, posY) tensors of shape [batchSize, height * width]
  static (Tensor, Tensor) createImagePositions({
    required int height,
    required int width,
    int batchSize = 1,
  }) {
    final List<int> posXList = [];
    final List<int> posYList = [];

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        posXList.add(x);
        posYList.add(y);
      }
    }

    Tensor posX = Tensor.from(posXList, [
      height * width,
    ], dataType: DataType.int64);
    Tensor posY = Tensor.from(posYList, [
      height * width,
    ], dataType: DataType.int64);

    if (batchSize > 1) {
      posX = posX.unsqueeze(0).expand([batchSize, height * width]);
      posY = posY.unsqueeze(0).expand([batchSize, height * width]);
    }

    return (posX, posY);
  }

  /// Create temporal position indices for a sequence
  ///
  /// Args:
  ///   seqLen: Sequence length
  ///   batchSize: Batch size
  ///
  /// Returns:
  ///   Position tensor of shape [batchSize, seqLen]
  static Tensor createTemporalPositions({
    required int seqLen,
    int batchSize = 1,
  }) {
    Tensor pos = Tensor.arange(0, seqLen, dataType: DataType.int64);

    if (batchSize > 1) {
      pos = pos.unsqueeze(0).expand([batchSize, seqLen]);
    }

    return pos;
  }
}
