import 'package:kamma/kamma.dart';
import 'package:kamma/src/gpt1/gpt1.dart';
import 'package:test/test.dart';

void main() {
  group('GPT1', () {
    test('OpenAIGPTLMHeadModel make and forward', () {
      final config = OpenAIGPTConfig(
        vocabSize: 100,
        nPositions: 16,
        nEmbd: 32,
        nLayer: 2,
        nHead: 4,
        afn: "gelu",
      );

      final model = OpenAIGPTLMHeadModel.make(config);

      final batchSize = 2;
      final seqLen = 8;

      // Create random input ids [batch, seqLen]
      final inputIds = Tensor.randint(config.vocabSize, [
        batchSize,
        seqLen,
      ], datatype: DataType.int64);

      final context = Context.best(isTraining: false);

      final logits = model.forward(inputIds, context: context);

      expect(logits.shape, equals([batchSize, seqLen, config.vocabSize]));

      // Verify weights are tied
      // Using dataPointer to check if they point to same memory?
      // Or just check values are same.
      // Modifying one should modify other.

      // But we just verify run for now.
    });
  });
}
