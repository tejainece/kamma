import 'package:tensor/tensor.dart';
import 'package:tensor/src/transformers/decision_transformer/decision_transformer_config.dart';
import 'package:tensor/src/transformers/decision_transformer/decision_transformer_model.dart';
import 'package:test/test.dart';

void main() {
  group('DecisionTransformer', () {
    late DecisionTransformerConfig config;
    late DecisionTransformerModel model;

    setUp(() {
      config = DecisionTransformerConfig(
        stateDim: 17,
        actDim: 4,
        maxEpLen: 100,
        nEmbd: 32,
        nLayer: 2,
        nHead: 4,
        vocabSize: 100,
        nPositions: 100,
      );
      model = DecisionTransformerModel.make(config: config, name: 'dt_test');
    });

    test('Initialization', () {
      expect(model, isNotNull);
      expect(model.config.stateDim, 17);
      expect(model.config.actDim, 4);
    });

    test('Forward Pass', () {
      final batchSize = 2;
      final seqLen = 10;
      final stateDim = config.stateDim;
      final actDim = config.actDim;

      final states = Tensor.randn([batchSize, seqLen, stateDim]);
      final actions = Tensor.randn([batchSize, seqLen, actDim]);
      final returns = Tensor.randn([batchSize, seqLen, 1]);
      final timesteps = Tensor.zeros([
        batchSize,
        seqLen,
      ], datatype: DataType.int64);
      final attentionMask = Tensor.ones(
        [batchSize, seqLen],
        datatype: DataType.float32,
        device: Device.cpu,
      );

      final context = Context(isTraining: false, device: Device.cpu);
      final output = model.forward(
        states,
        actions,
        returns,
        timesteps,
        attentionMask: attentionMask,
        context: context,
      );

      expect(output.statePreds.shape, [batchSize, seqLen, stateDim]);
      expect(output.actionPreds.shape, [batchSize, seqLen, actDim]);
      expect(output.returnPreds.shape, [batchSize, seqLen, 1]);
    });
  });
}
