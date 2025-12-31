import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

void main() {
  group('GPT2AttentionMethod.makeCausalMask', () {
    late Device device;
    late Dropout dropout;

    setUp(() {
      device = Device.cpu;
      dropout = Dropout(0.1);
    });

    test('Eager method returns float mask', () {
      final method = GPT2AttentionMethod.make(
        GPT2AttentionMethodType.eager,
        scaleFactor: 1.0,
        isCausal: true,
        maxPositionEmbeddings: 1024,
        attnDropout: dropout,
      );

      final maskResult = method.makeCausalMask(
        1, // batchSize
        4, // seqLength
        DataType.float32,
        device,
      );

      expect(maskResult, isA<SimpleCausalMask>());
      final mask = (maskResult as SimpleCausalMask).mask;

      expect(mask.shape, [1, 1, 4, 4]);
      // Verify float values: 0.0 for keep, min for mask
      final data = mask.view([-1]).toList();
      final minVal = DataType.float32.fInfo.min;

      // Keep (0.0) vs Mask (minVal)
      // Row 0: 0, min, min, min
      expect(data[0], 0.0);
      expect(data[1], minVal);
    });

    test('SDPA method returns boolean mask', () {
      final method = GPT2AttentionMethod.make(
        GPT2AttentionMethodType.sdap,
        scaleFactor: 1.0,
        isCausal: true,
        maxPositionEmbeddings: 1024,
        attnDropout: dropout,
      );

      // Pass attention mask to force creation (otherwise returns null for optimization)
      final dummyMask = Tensor.ones(
        [1, 4],
        device: device,
        dataType: DataType.boolean,
      );
      final maskResult = method.makeCausalMask(
        1,
        4,
        DataType.float32,
        device,
        attentionMask: dummyMask,
      );

      expect(maskResult, isA<SimpleCausalMask>());
      final mask = (maskResult as SimpleCausalMask).mask;

      // SDPA returns boolean mask
      expect(mask.dataType, DataType.boolean);

      // Logic: True where we keep?
      // Our implementation: floatMask.eq(0.0) -> True where keep.
      // Row 0: True, False, False, False

      // Note: Tensor.toList for boolean might not be supported. Cast to Int.
      final data = mask.to(dataType: DataType.int32).view([-1]).toList();

      expect(data[0], 1); // Keep (True -> 1)
      expect(data[1], 0); // Mask (False -> 0)
    });

    test('Flash Attention returns correct structure', () {
      final method = GPT2AttentionMethod.make(
        GPT2AttentionMethodType.flashAttention2,
        scaleFactor: 1.0,
        isCausal: true,
        maxPositionEmbeddings: 1024,
        attnDropout: dropout,
      );

      // Null attention mask -> Null result (optimized)
      final maskResult = method.makeCausalMask(1, 4, DataType.float32, device);
      expect(maskResult, isNull);

      // With attention mask -> Returns mask (stubbed)
      final dummyMask = Tensor.ones(
        [1, 4],
        device: device,
        dataType: DataType.boolean,
      );
      final maskResultWithAttr = method.makeCausalMask(
        1,
        4,
        DataType.float32,
        device,
        attentionMask: dummyMask,
      );
      expect(maskResultWithAttr, isNotNull);
      expect(maskResultWithAttr, isA<SimpleCausalMask>());
    });

    test('Flex Attention returns BlockCausalMask', () {
      final method = GPT2AttentionMethod.make(
        GPT2AttentionMethodType.flexAttention,
        scaleFactor: 1.0,
        isCausal: true,
        maxPositionEmbeddings: 1024,
        attnDropout: dropout,
      );

      final maskResult = method.makeCausalMask(1, 4, DataType.float32, device);
      expect(maskResult, isA<BlockCausalMask>());
    });
  });
}
