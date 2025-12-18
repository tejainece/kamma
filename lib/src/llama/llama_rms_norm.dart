import 'package:tensor/tensor.dart';
import 'llama_config.dart';

class LlamaRMSNorm extends RMSNorm {
  LlamaRMSNorm(LlamaConfig config)
    : super([config.hiddenSize], eps: config.rmsNormEps);

  // You might want to override forward if LlamaRMSNorm logic differed, but it seems compatible.
  // transformers: hidden_states * torch.rsqrt(variance + self.variance_epsilon) * weight
  // tensor: NNUtil.rmsNorm which typically does the same.

  // If we need to support loading from config-based construction, this is enough.
  // Note: RMSNorm in tensor.dart handles weight/parameters.
}
