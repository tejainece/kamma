import 'package:tensor/tensor.dart';
import 'deepseek_config.dart';

class DeepSeekRMSNorm extends RMSNorm {
  DeepSeekRMSNorm(DeepSeekConfig config, {int? dim})
    : super([dim ?? config.hiddenSize], eps: config.rmsNormEps);
}
