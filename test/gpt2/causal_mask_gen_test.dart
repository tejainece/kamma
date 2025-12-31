import 'package:kamma/kamma.dart';
import 'package:test/test.dart';
import 'package:path/path.dart' as p;

void main() {
  group('GPT2 Causal Mask Generation Test', () {
    late Map<String, Tensor> tensors;
    late Device device;
    late SafeTensorLoader loader;

    setUpAll(() async {
      device = Device.cpu;
      final path = p.join(
        '..',
        'testdata',
        'test_data',
        'llm',
        'gpt2',
        'causal_mask_eager.safetensors',
      );

      print('Loading safetensors from $path');
      final file = await SafeTensorsFile.load(path);
      loader = file.mmapTensorLoader();

      tensors = {};
      for (final key in file.header.tensorInfos.keys) {
        tensors[key] = await loader.loadByName(key, device: device);
      }
      print('Loaded ${tensors.length} tensors');
    });

    test('Verify Eager Causal Masks', () {
      final method = GPT2AttentionMethod.make(
        GPT2AttentionMethodType.eager,
        scaleFactor: 1.0,
        isCausal: true,
        maxPositionEmbeddings: 1024,
        attnDropout: Dropout(0.0),
      );

      tensors.forEach((key, expectedMask) {
        // Parse key: cnt_b{b}_s{s}_p{p}_mask{Type}
        final parts = key.split('_');
        final b = int.parse(parts[1].substring(1)); // bX
        final s = int.parse(parts[2].substring(1)); // sX
        final p = int.parse(parts[3].substring(1)); // pX
        final type = parts[4]; // maskNone or mask2d

        Tensor? attentionMask;
        if (type == 'mask2d') {
          // Recreate the logic from python script
          // kv_length = s + p
          final kvLength = s + p;

          // Eager impl expects prepared Additive mask (0.0 for keep, min_float for mask).
          // Python script used 1s (keep) and 0s (mask).
          // So we convert:
          // 1 -> 0.0
          // 0 -> -3.4e38 (float32 min approx)

          final batchData = List.filled(b * kvLength, 0.0); // Default keep

          if (kvLength > 1) {
            // We want masked (0 in python) to be min_float.
            // Python set index 0 to masked.
            for (int i = 0; i < b; i++) {
              batchData[i * kvLength] = -3.4028234663852886e+38;
            }
          }

          // Expand to 4D: (batch, 1, 1, kv_length)
          attentionMask = Tensor.from(
            batchData,
            [b, 1, 1, kvLength],
            dataType: DataType.float32,
            device: device,
          );
        }

        final result = method.makeCausalMask(
          b,
          s,
          expectedMask.dataType,
          device,
          pastKeyValuesLength: p,
          attentionMask: attentionMask,
        );

        expect(result, isNotNull, reason: 'Key: $key');
        final resultTensor = (result as SimpleCausalMask).mask;

        // Verify shape
        expect(
          resultTensor.shape,
          expectedMask.shape,
          reason: 'Shape mismatch for $key',
        );

        // Verify values
        final diff = (resultTensor - expectedMask).abs().max();
        final maxDiff = diff.scalar as double;

        // Use a slightly larger tolerance for float comparison of large values
        expect(
          maxDiff,
          lessThan(1e-4),
          reason: 'Value mismatch for $key (max diff: $maxDiff)',
        );
      });
    });
  });
}
