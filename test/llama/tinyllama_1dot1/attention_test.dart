import 'dart:io';
import 'package:tensor/tensor.dart';
import 'package:kamma/src/llama/llama_attention.dart';
import 'package:test/test.dart';

class TestCase {
  final String name;
  final int numHeads;
  final double ropeTheta;
  final int layerIdx;
  final int maxPositionEmbeddings;
  final Tensor hiddenStates;
  final Tensor attentionMask;
  final Tensor positionIds;
  final Tensor cos;
  final Tensor sin;
  final Tensor expectedOutput;

  TestCase({
    required this.name,
    required this.numHeads,
    required this.ropeTheta,
    required this.layerIdx,
    required this.maxPositionEmbeddings,
    required this.hiddenStates,
    required this.attentionMask,
    required this.positionIds,
    required this.cos,
    required this.sin,
    required this.expectedOutput,
  });

  static Future<TestCase> load(SafeTensorLoader loader, String name) async {
    final device = Device.cpu;
    final metadata = loader.metadata;
    final hiddenStates = await loader.loadByName(
      '$name.hidden_states',
      device: device,
    );
    final attentionMask = await loader.loadByName(
      '$name.attention_mask',
      device: device,
    );
    final positionIds = await loader.loadByName(
      '$name.position_ids',
      device: device,
    );
    final cos = await loader.loadByName('$name.cos', device: device);
    final sin = await loader.loadByName('$name.sin', device: device);
    final expectedOutput = await loader.loadByName(
      '$name.output',
      device: device,
    );
    return TestCase(
      name: name,
      numHeads: int.parse(metadata['$name.num_heads']!),
      ropeTheta: double.parse(metadata['$name.rope_theta']!),
      layerIdx: int.parse(metadata['$name.layer_idx']!),
      maxPositionEmbeddings: int.parse(
        metadata['$name.max_position_embeddings']!,
      ),
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      positionIds: positionIds,
      cos: cos,
      sin: sin,
      expectedOutput: expectedOutput,
    );
  }
}

void main() {
  group('LlamaAttention TinyLlama 1.1B', () {
    late SafeTensorsFile file;
    late SafeTensorLoader loader;

    setUpAll(() async {
      final path =
          './testdata/test_data/llm/llama/tinyllama_1dot1/attention_test.safetensors';
      if (!File(path).existsSync()) {
        throw Exception(
          'Test data not found. Run python test generation script first.',
        );
      }
      file = await SafeTensorsFile.load(path);
      loader = file.mmapTensorLoader();
    });

    test('forward pass matches PyTorch', () async {
      final seen = <String>{};
      for (final key in loader.tensorInfos.keys) {
        final name = key.split('.').first;
        if (seen.contains(name)) continue;
        seen.add(name);

        final test = await TestCase.load(loader, name);

        // Load attention module
        final attention = await LlamaAttention.loadFromSafeTensor(
          loader,
          prefix: '$name.self_attn.',
          name: 'self_attn',
          layerIdx: test.layerIdx,
          numHeads: test.numHeads,
          maxPositionEmbeddings: test.maxPositionEmbeddings,
          ropeTheta: test.ropeTheta,
          attentionDropoutProb: 0.0,
          isCausal: true,
        );

        final context = Context(device: Device.cpu);

        final output = attention.forward(
          test.hiddenStates,
          context: context,
          attentionMask: test.attentionMask,
          positionIds: test.positionIds,
          positionEmbeddings: (cos: test.cos, sin: test.sin),
        );

        final close = output.allClose(
          test.expectedOutput,
          atol: 1e-4,
          rtol: 1e-4,
        );
        if (!close) {
          final diff = (output - test.expectedOutput).abs();
          final maxDiff = diff.max().scalar as double;
          print('Max difference: $maxDiff');
        }
        expect(
          close,
          isTrue,
          reason: '$name: Output does not match PyTorch output',
        );
      }
    });
  });
}
