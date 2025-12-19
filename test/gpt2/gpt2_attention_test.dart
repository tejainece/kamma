import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

class Test {
  final String name;
  final Tensor inputEmbeddings;
  final Tensor? attentionMask;
  final Tensor expectedOutputEmbeddings;
  final Tensor expectedAttentionWeights;

  final GPT2Attention attention;

  Test({
    required this.name,
    required this.inputEmbeddings,
    required this.attentionMask,
    required this.expectedOutputEmbeddings,
    required this.expectedAttentionWeights,
    required this.attention,
  });

  static Future<Test> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final inputEmbeddings = await loader.loadByName(
      '$name.inputEmbeddings',
      device: device,
    );
    final expectedOutputEmbeddings = await loader.loadByName(
      '$name.outputEmbeddings',
      device: device,
    );
    final expectedAttentionWeights = await loader.loadByName(
      '$name.attentionWeights',
      device: device,
    );
    final attentionMask = await loader.tryLoadByName(
      '$name.attentionMask',
      device: device,
    );

    final numHeads = int.parse(loader.header.metadata['$name.numHeads']!);
    final layerIdx = int.parse(loader.header.metadata['$name.layerIdx']!);
    final maxPositionEmbeddings = int.parse(
      loader.header.metadata['$name.maxPositionEmbeddings']!,
    );
    final scaleAttnByInverseLayerIdx =
        loader.header.metadata['$name.scaleAttnByInverseLayerIdx'] == 'true';

    final attention = await GPT2Attention.loadFromSafeTensor(
      loader,
      prefix: '$name.',
      name: name,
      layerIdx: layerIdx,
      numHeads: numHeads,
      scaleAttnByInverseLayerIdx: scaleAttnByInverseLayerIdx,
      maxPositionEmbeddings: maxPositionEmbeddings,
      attnFuncType: GPT2AttentionMethodType.eager,
      isCrossAttention: false,
      attentionDropoutProbability: 0,
      residualDropoutProbability: 0,
    );
    return Test(
      name: name,
      inputEmbeddings: inputEmbeddings,
      expectedOutputEmbeddings: expectedOutputEmbeddings,
      expectedAttentionWeights: expectedAttentionWeights,
      attentionMask: attentionMask,
      attention: attention,
    );
  }

  static List<String> getTestNames(SafeTensorLoader loader) {
    final names = <String>[];
    for (final name in loader.tensorInfos.keys) {
      if (name.endsWith('.inputEmbeddings')) {
        names.add(name.split('.')[0]);
      }
    }
    return names;
  }
}

void main() {
  group('GPT2Attention', () {
    final testDataFiles = [
      'testdata/test_data/llm/gpt2/gpt2_attention.safetensors',
    ];
    final loaders = <SafeTensorLoader>[];

    setUpAll(() async {
      for (final fileName in testDataFiles) {
        final file = await SafeTensorsFile.load(fileName);
        final loader = file.mmapTensorLoader();
        loaders.add(loader);
      }
    });

    test('forward pass matches pytorch', () async {
      final context = Context.best();

      for (final loader in loaders) {
        for (final name in Test.getTestNames(loader)) {
          final test = await Test.loadFromSafeTensor(
            loader,
            name,
            device: context.device,
          );

          final result = test.attention.forward(
            test.inputEmbeddings,
            attentionMask: test.attentionMask,
            context: context,
          );

          final output = result.outputEmbeddings.to(device: context.device);
          final diff =
              (output - test.expectedOutputEmbeddings).abs().max().scalar
                  as double;
          print('Max difference: $diff');
          expect(diff, closeTo(0.0, 1e-4));
        }
      }

      print('\n✓ Successfully passed GPT2Attention testcases');
    });
  });
}
