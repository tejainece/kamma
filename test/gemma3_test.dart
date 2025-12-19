import 'package:kamma/kamma.dart';
import 'package:kamma/src/gemma3/gemma3_config.dart';
import 'package:kamma/src/gemma3/gemma3_model.dart';
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Gemma3ForCausalLM', () {
    late Gemma3Config config;
    late Gemma3ForCausalLM model;
    late Context context;

    setUp(() {
      config = Gemma3Config(
        vocabSize: 100, // Small vocab for testing
        hiddenSize: 64,
        intermediateSize: 128,
        numHiddenLayers: 2,
        numAttentionHeads: 4,
        numKeyValueHeads: 2,
        headDim: 16,
        ropeTheta: 10000.0,
        rmsNormEps: 1e-6,
        hiddenActivation: 'gelu',
        padTokenId: 0,
        bosTokenId: 1,
        eosTokenId: 2,
        maxPositionEmbeddings: 512,
        finalLogitSoftcapping: 30.0,
        attnLogitSoftcapping: 50.0,
        queryPreAttnScalar: 256.0,
        attentionBias: false,
        layerTypes: ['sliding_attention', 'sliding_attention'], // Match layers
        slidingWindow: 128,
      );

      model = Gemma3ForCausalLM.make(name: 'gemma3_test', config: config);
      context = Context(device: Device.cpu);
    });

    test('initialization', () {
      expect(model.model.layers.length, equals(2));
      expect(model.config.hiddenSize, equals(64));
    });

    test('forward pass without cache', () {
      final batchSize = 2;
      final seqLen = 10;
      final inputIds = Tensor.randint(
        config.vocabSize,
        [batchSize, seqLen],
        datatype: DataType.int64,
        device: Device.cpu,
      );
      final attentionMask = Tensor.ones(
        [batchSize, 1, 1, seqLen],
        dataType: DataType.float32,
        device: Device.cpu,
      );

      final outputs = model.forward(
        inputIds,
        context: context,
        attentionMask: attentionMask,
        useCache: false,
      );

      // Check logits shape: [batchSize, seqLen, vocabSize]
      expect(
        outputs.logits.shape,
        equals([batchSize, seqLen, config.vocabSize]),
      );
      expect(outputs.pastKeyValues, isNull);
    });

    test('forward pass with cache', () {
      final batchSize = 1;
      final seqLen = 5;
      final inputIds = Tensor.randint(
        config.vocabSize,
        [batchSize, seqLen],
        datatype: DataType.int64,
        device: Device.cpu,
      );

      // First run to get cache
      final outputs1 = model.forward(
        inputIds,
        context: context,
        useCache: true,
      );

      expect(outputs1.pastKeyValues, isNotNull);
      expect(outputs1.pastKeyValues!.length, equals(config.numHiddenLayers));
      // Each layer has [k, v]
      expect(outputs1.pastKeyValues![0].length, equals(2));
      // Check K shape: [B, H_kv, S, D_head]
      // S should be seqLen
      expect(
        outputs1.pastKeyValues![0][0].shape,
        equals([batchSize, config.numKeyValueHeads, seqLen, config.headDim]),
      );

      // Second run with new token and past cache
      final nextInputIds = Tensor.randint(
        config.vocabSize,
        [batchSize, 1],
        datatype: DataType.int64,
        device: Device.cpu,
      );

      final outputs2 = model.forward(
        nextInputIds,
        context: context,
        pastKeyValues: outputs1.pastKeyValues!,
        useCache: true,
      );

      expect(outputs2.logits.shape, equals([batchSize, 1, config.vocabSize]));
      expect(outputs2.pastKeyValues, isNotNull);
      // New cache length should be seqLen + 1
      expect(outputs2.pastKeyValues![0][0].shape[2], equals(seqLen + 1));
    });
  });
}
