import 'package:kamma/kamma.dart';
import 'package:kamma/src/deepseek/deepseek_config.dart';
import 'package:kamma/src/deepseek/deepseek_model.dart';
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('DeepSeek R1', () {
    late DeepSeekConfig config;

    setUp(() {
      config = DeepSeekConfig(
        vocabSize: 100,
        hiddenSize: 64,
        intermediateSize: 128,
        numHiddenLayers: 2,
        numAttentionHeads: 4,
        numKeyValueHeads: 4,
        nRoutedExperts: 4,
        nSharedExperts: 1,
        topK: 2,
        moeIntermediateSize: 32,
        maxPositionEmbeddings: 128,
        qLoraRank: 32,
        kvLoraRank: 32,
        nopeHeadDim: 8,
        ropeHeadDim: 8,
      );
    });

    test('DeepSeekForCausalLM Initialization and Forward Pass', () {
      final model = DeepSeekForCausalLM.make(config);

      final inputIds = Tensor.from(
        [1, 2, 3, 4],
        [1, 4],
        dataType: DataType.int32,
      );
      final context = Context(device: Device.cpu, isTraining: false);

      final output = model.forward(inputIds, context: context);

      expect(output.shape, [1, 4, 100]); // (Batch, Seq, Vocab)
      print("Output shape verified");
    });
  });
}
