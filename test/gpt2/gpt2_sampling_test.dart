import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('Tensor Sampling Operations', () {
    test('topk', () {
      final t = Tensor.from(
        [1.0, 5.0, 2.0, 8.0, 3.0],
        [5],
        datatype: DataType.float64,
      );
      final (values, indices) = t.topk(3);

      expect(values.shape, [3]);
      expect(indices.shape, [3]);

      final valuesList = values.toList();
      final indicesList = indices.toList();

      expect(valuesList, [8.0, 5.0, 3.0]);
      expect(indicesList, [3, 1, 4]);
    });

    test('sort', () {
      final t = Tensor.from(
        [1.0, 5.0, 2.0, 8.0, 3.0],
        [5],
        datatype: DataType.float64,
      );
      final (values, indices) = t.sort(descending: true);

      expect(values.toList(), [8.0, 5.0, 3.0, 2.0, 1.0]);
      expect(indices.toList(), [3, 1, 4, 2, 0]);
    });

    test('cumsum', () {
      final t = Tensor.from(
        [1.0, 2.0, 3.0, 4.0],
        [4],
        datatype: DataType.float64,
      );
      final result = t.cumsum(0);

      expect(result.toList(), [1.0, 3.0, 6.0, 10.0]);
    });

    test('multinomial', () {
      final weights = Tensor.from(
        [0.0, 10.0, 0.0, 0.0],
        [4],
        datatype: DataType.float64,
      ); // Only index 1 has probability
      final result = weights.multinomial(5, replacement: true);

      expect(result.shape, [5]);
      final resultList = result.toList();
      for (final idx in resultList) {
        expect(idx, 1);
      }
    });
  });
}
