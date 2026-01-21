import 'package:kamma/kamma.dart';
import 'package:test/test.dart';
import 'package:kamma/src/common/rope/rope.dart';

class Test {
  final int dim;
  final double base;
  final double maxPositionEmbeddings;
  final Map<String, dynamic> ropeScaling;
  final Tensor invFreq;

  Test({
    required this.dim,
    required this.base,
    required this.maxPositionEmbeddings,
    required this.ropeScaling,
    required this.invFreq,
  });

  static Future<Test> load(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final dim = int.parse(loader.metadata['$name.dim']!);
    final base = double.parse(loader.metadata['$name.base']!);
    final maxPositionEmbeddings = double.parse(
      loader.metadata['$name.max_position_embeddings']!,
    );

    final ropeScaling = <String, dynamic>{};
    // Extract rope scaling params from metadata
    final keys = [
      'factor',
      'low_freq_factor',
      'high_freq_factor',
      'original_max_position_embeddings',
      'rope_type',
    ];
    for (final key in keys) {
      if (loader.metadata.containsKey('$name.$key')) {
        var val = loader.metadata['$name.$key']!;
        if (num.tryParse(val) != null) {
          ropeScaling[key] = num.parse(val);
        } else {
          ropeScaling[key] = val;
        }
      }
    }

    final invFreq = await loader.loadByName('$name.inv_freq', device: device);
    return Test(
      dim: dim,
      base: base,
      maxPositionEmbeddings: maxPositionEmbeddings,
      ropeScaling: ropeScaling,
      invFreq: invFreq,
    );
  }
}

void main() {
  group('Llama3RopeArgs', () {
    late SafeTensorsFile file;
    late SafeTensorLoader loader;

    setUpAll(() async {
      final path =
          'testdata/test_data/llm/_common/rope/llama3_rope_args.safetensors';
      file = await SafeTensorsFile.load(path);
      loader = file.cpuLoader();
    });

    test('compute matches generated values', () async {
      final seen = <String>{};
      for (final key in loader.tensorInfos.keys) {
        final name = key.split('.').first;
        if (seen.contains(name)) continue;
        seen.add(name);

        final tc = await Test.load(loader, name, device: Device.cpu);

        final args = Llama3RopeArgs.fromMap(tc.ropeScaling);

        final result = args.compute(
          tc.dim,
          tc.base,
          tc.maxPositionEmbeddings.toInt(),
        );

        expect(result.attentionScaling, equals(1.0));

        final diff = (result.invFreq - tc.invFreq).abs().max().scalar as double;
        print('Test case $name: diff=$diff');
        expect(
          diff,
          lessThan(1e-5),
          reason: 'dim=${tc.dim}, base=${tc.base}, scaling=${tc.ropeScaling}',
        );
      }
    });
  });
}
