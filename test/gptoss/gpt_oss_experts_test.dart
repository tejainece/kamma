import 'package:kamma/src/gpt_oss/gpt_oss_moe.dart';
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  group('GptOssExperts', () {
    test('GptOssExperts forward', () {
      final numExperts = 4;
      final embedDim = 8;
      final innerDim = 16;
      final seqLen = 2;
      final batchSize = 1;

      // Create random tensors for experts
      final gateUp = Tensor.randn([numExperts, embedDim, innerDim * 2]);
      final down = Tensor.randn([numExperts, innerDim, embedDim]);

      final experts = GptOssExperts(
        name: 'experts',
        gateUpProj: gateUp,
        downProj: down,
      );

      final x = Tensor.randn([batchSize, seqLen, embedDim]);
      final context = Context.best();

      // Move input to device
      final xDevice = x.to(device: context.device);

      // Mock routing weights (dense, equal)
      Tensor routingWeights =
          Tensor.ones([batchSize, seqLen, numExperts]) * (1.0 / numExperts);
      routingWeights = routingWeights.to(device: context.device);

      final y = experts.forward(
        xDevice,
        routerIndices: Tensor.zeros(
          [1],
          dataType: DataType.int32,
          device: context.device,
        ),
        routingWeights: routingWeights,
        context: context,
      );

      expect(y.shape, equals([batchSize, seqLen, embedDim]));
    });
  });
}
