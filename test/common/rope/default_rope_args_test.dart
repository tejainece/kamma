import 'package:kamma/kamma.dart';
import 'package:test/test.dart';
import 'package:kamma/src/common/rope/rope.dart';
import 'package:universal_io/io.dart';

class Test {
  final int dim;
  final double base;
  final Tensor invFreq;

  Test({required this.dim, required this.base, required this.invFreq});

  static Future<Test> load(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final dim = int.parse(loader.metadata['$name.dim']!);
    final base = double.parse(loader.metadata['$name.base']!);
    final invFreq = await loader.loadByName('$name.inv_freq', device: device);
    return Test(dim: dim, base: base, invFreq: invFreq);
  }
}

void main() {
  group('DefaultRopeArgs', () {
    late SafeTensorsFile file;
    late SafeTensorLoader loader;

    setUpAll(() async {
      final path =
          'testdata/test_data/llm/_common/rope/default_rope_args.safetensors';
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
        final result = DefaultRopeArgs.instance.compute(tc.dim, tc.base);

        expect(result.attentionScaling, equals(1.0));

        final diff = (result.invFreq - tc.invFreq).abs().max().scalar as double;
        expect(diff, lessThan(1e-5), reason: 'dim=${tc.dim}, base=${tc.base}');
      }
    });
  });
}
