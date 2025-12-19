import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

void main() {
  test('GPT2LMHeadModel forward pass', () {
    final config = GPT2Config(
      vocabSize: 100,
      nPositions: 20,
      embedDim: 32,
      numLayers: 2,
      numHeads: 4,
    );

    final model = GPT2LMHeadModel.make(config: config, name: 'gpt2');

    final batchSize = 2;
    final seqLength = 10;

    final inputIds = Tensor.ones([1, 10], dataType: DataType.int64);
    // final inputIds = (Tensor.rand([1, 10]) * config.vocabSize).to(dataType: DataType.int64);

    // Create dummy attention mask
    // final attentionMask = Tensor.ones([1, 10]);

    // Create dummy position ids
    final positionIds = Tensor.arange(
      0,
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
    final config = GPT2Config(embedDim: 32, numHeads: 4, numLayers: 2);
    final attention = GPT2Attention.make(
      name: 'attn',
      layerIdx: 0,
      embedDim: config.embedDim,
      numHeads: config.numHeads,
      attentionDropoutProbability: config.attentionDropoutProbability,
      residualDropoutProbability: config.residualDropoutProbability,
      isCrossAttention: false,
      scaleAttnByInverseLayerIdx: config.scaleAttnByInverseLayerIdx,
      maxPositionEmbeddings: config.maxPositionEmbeddings,
    );
    final context = Context.best();

    final batchSize = 2;
    final seqLength = 10;
    final hiddenStates = Tensor.randn([batchSize, seqLength, config.embedDim]);

    final output = attention.forward(hiddenStates, context: context);

    expect(output.outputEmbeddings.shape, [
      batchSize,
      seqLength,
      config.embedDim,
    ]);
    expect(output.attentionWeights.shape, [
      batchSize,
      config.numHeads,
      seqLength,
      seqLength,
    ]);
  });

  test('GPT2MLP forward pass', () {
    final config = GPT2Config(embedDim: 32, mlpInnerDim: 64);
    final mlp = GPT2MLP.make(
      embedDim: config.embedDim,
      mlpInnerDim: config.mlpInnerDim,
      residualDropoutProbability: config.residualDropoutProbability,
      activation: Activation.geluTanh,
      name: 'mlp',
    );
    final context = Context.best();

    final batchSize = 2;
    final seqLength = 10;
    final hiddenStates = Tensor.randn([batchSize, seqLength, config.embedDim]);

    final output = mlp.forward(hiddenStates, context: context);

    expect(output.shape, [batchSize, seqLength, config.embedDim]);
  });
}
