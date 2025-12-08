import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

void main() {
  final context = Context.best();
  group('GptOssForCausalLM', () {
    late GptOssConfig config;

    setUp(() {
      config = GptOssConfig(
        vocabSize: 100,
        embedDim: 32,
        nLayer: 2,
        nHead: 4,
        nPositions: 20,
        numExperts: 4,
        numExpertsPerToken: 2, // Top-2
        ropeTheta: 10000.0,
      );
    });

    test('forward pass shapes', () {
      final model = GptOssForCausalLM.make(config: config, name: 'model');

      final batchSize = 2;
      final seqLen = 10;
      final inputIds = Tensor.randint(config.vocabSize, [
        batchSize,
        seqLen,
      ], datatype: DataType.int64).to(device: context.device);

      final output = model.forward(inputIds, context: context);

      expect(output.shape, [batchSize, seqLen, config.vocabSize]);

      // Check generation runs
      final generated = model.generate(
        inputIds,
        maxNewTokens: 5,
        context: context,
      );

      expect(generated.shape, [batchSize, seqLen + 5]);
    });
  });

  group('GptOssMoE', () {
    test('forward pass shape', () {
      final config = GptOssConfig(
        embedDim: 32,
        nHead: 4,
        numExperts: 4,
        numExpertsPerToken: 2,
      );
      final moe = GptOssMoE.make(
        name: 'moe',
        embedDim: config.embedDim,
        nInner: 4 * config.embedDim,
        numExperts: config.numExperts,
        numExpertsPerToken: config.numExpertsPerToken!,
      );
      final input = Tensor.randn([2, 5, 32]).to(device: context.device);
      final output = moe.forward(input, context: context);
    });
  });

  group('GptOssAttention', () {
    test('forward pass shape (with GQA and RoPE)', () {
      final attn = GptOssAttention.make(
        name: 'attn',
        isCrossAttention: false,
        layerIdx: 0,
        embedDim: 32,
        numHeads: 4,
        nPositions: 20,
        numKeyValueHeads: 2,
        ropeTheta: 10000.0,
        residDropoutP: 0.0,
        attentionDropoutP: 0.0,
      );
      final input = Tensor.randn([2, 5, 32]).to(device: context.device);
      final output = attn.forward(input, context: context);
      expect(output.shape, [2, 5, 32]);
    });
  });
}
