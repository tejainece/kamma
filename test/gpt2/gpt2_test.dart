import 'package:test/test.dart';
import 'package:tensor/tensor.dart';

void main() {
  test('GPT2LMHeadModel forward pass', () {
    final config = GPT2Config(
      vocabSize: 100,
      nPositions: 20,
      nEmbd: 32,
      nLayer: 2,
      nHead: 4,
    );

    final model = GPT2LMHeadModel.make(config: config, name: 'gpt2');

    final batchSize = 2;
    final seqLength = 10;

    final inputIds = Tensor.ones([1, 10], datatype: DataType.int64);
    // final inputIds = (Tensor.rand([1, 10]) * config.vocabSize).to(dataType: DataType.int64);

    // Create dummy attention mask
    // final attentionMask = Tensor.ones([1, 10]);

    // Create dummy position ids
    final positionIds = Tensor.arange(
      10,
      datatype: DataType.int64,
    ).unsqueeze(0).expand([batchSize, seqLength]);

    final context = Context.best();

    final output = model.forward(
      inputIds,
      positionIds: positionIds,
      context: context,
    );

    expect(output.shape, [batchSize, seqLength, config.vocabSize]);
  });

  test('GPT2Attention forward pass', () {
    final config = GPT2Config(nEmbd: 32, nHead: 4, nLayer: 2);
    final attention = GPT2Attention.make(config: config, name: 'attn');
    final context = Context.best();

    final batchSize = 2;
    final seqLength = 10;
    final hiddenStates = Tensor.randn([batchSize, seqLength, config.nEmbd]);

    final output = attention.forward(hiddenStates, context: context);

    expect(output.shape, [batchSize, seqLength, config.nEmbd]);
  });

  test('GPT2MLP forward pass', () {
    final config = GPT2Config(nEmbd: 32, nInner: 64);
    final mlp = GPT2MLP.make(config: config, name: 'mlp');
    final context = Context.best();

    final batchSize = 2;
    final seqLength = 10;
    final hiddenStates = Tensor.randn([batchSize, seqLength, config.nEmbd]);

    final output = mlp.forward(hiddenStates, context: context);

    expect(output.shape, [batchSize, seqLength, config.nEmbd]);
  });
}
