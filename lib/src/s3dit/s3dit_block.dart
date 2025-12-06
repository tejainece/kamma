import 'package:tensor/tensor.dart';
import 'package:tensor/src/transformers/s3dit/s3dit_config.dart';
import 'package:tensor/src/transformers/s3dit/s3dit_attention.dart';
import 'package:tensor/src/transformers/s3dit/s3dit_mlp.dart';

/// Single S3-DiT transformer block with sandwich-style RMSNorm
///
/// Architecture:
/// - Pre-normalization (RMSNorm) before attention
/// - Self-attention with residual connection
/// - Pre-normalization (RMSNorm) before MLP
/// - MLP with residual connection
class S3DiTBlock extends Module {
  final RMSNorm norm1;
  final S3DiTAttention attention;
  final RMSNorm norm2;
  final S3DiTMLP mlp;

  S3DiTBlock({
    required super.name,
    required this.norm1,
    required this.attention,
    required this.norm2,
    required this.mlp,
  });

  @override
  Tensor forward(
    Tensor hiddenStates, {
    Tensor? posX,
    Tensor? posY,
    Tensor? posT,
    Tensor? attentionMask,
    required Context context,
  }) {
    context.onloadModule(this);

    // Pre-norm + attention + residual
    Tensor residual = hiddenStates;
    hiddenStates = norm1.forward(hiddenStates, context: context);
    hiddenStates = attention.forward(
      hiddenStates,
      posX: posX,
      posY: posY,
      posT: posT,
      attentionMask: attentionMask,
      context: context,
    );
    hiddenStates = hiddenStates + residual;

    // Pre-norm + MLP + residual
    residual = hiddenStates;
    hiddenStates = norm2.forward(hiddenStates, context: context);
    hiddenStates = mlp.forward(hiddenStates, context: context);
    hiddenStates = hiddenStates + residual;

    return hiddenStates;
  }

  @override
  void resetParameters() {
    norm1.resetParameters();
    attention.resetParameters();
    norm2.resetParameters();
    mlp.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...norm1.parameters,
    ...attention.parameters,
    ...norm2.parameters,
    ...mlp.parameters,
  ];

  @override
  late final Iterable<Module> submodules = [norm1, attention, norm2, mlp];

  @override
  Map<String, dynamic> get meta => {};

  static S3DiTBlock make({required S3DiTConfig config, required String name}) {
    final norm1 = RMSNorm.make(
      name: 'norm1',
      normalizedShape: [config.hiddenSize],
      eps: config.layerNormEps,
    );

    final attention = S3DiTAttention.make(config: config, name: 'attn');

    final norm2 = RMSNorm.make(
      name: 'norm2',
      normalizedShape: [config.hiddenSize],
      eps: config.layerNormEps,
    );

    final mlp = S3DiTMLP.make(config: config, name: 'mlp');

    return S3DiTBlock(
      name: name,
      norm1: norm1,
      attention: attention,
      norm2: norm2,
      mlp: mlp,
    );
  }
}
