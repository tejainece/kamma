import 'package:kamma/kamma.dart';

/// Creates a causal mask of size [batchSize, 1, seqLength, seqLength] or [batchSize, 1, seqLength, seqLength + pastKeyValuesLength].
/// The mask contains min values for positions that should be masked and 0 for positions that should be kept.
/// Creates a causal mask using the provided [method].
/// This acts as a dispatcher similar to transformers library's `create_causal_mask`.
Tensor? createCausalMask(
  int batchSize,
  int seqLength,
  DataType dataType,
  Device device, {
  required GPT2AttentionMethod method,
  int pastKeyValuesLength = 0,
  Tensor? attentionMask, // (batchSize, seqLength)
}) {
  final causalMaskObj = method.makeCausalMask(
    batchSize,
    seqLength,
    dataType,
    device,
    pastKeyValuesLength: pastKeyValuesLength,
    attentionMask: attentionMask,
  );

  if (causalMaskObj == null) {
    return null;
  }

  if (causalMaskObj is SimpleCausalMask) {
    return causalMaskObj.mask;
  } else if (causalMaskObj is BlockCausalMask) {
    // TODO: Handle BlockMask when supported
    return causalMaskObj.blockMask;
  }

  throw UnimplementedError(
    'Unknown CausalMask type: ${causalMaskObj.runtimeType}',
  );
}
