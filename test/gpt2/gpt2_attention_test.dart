import 'package:kamma/kamma.dart';
import 'package:kamma/src/gpt2/gpt2_attention.dart';
import 'package:test/test.dart';
import 'package:tensor/tensor.dart';
import 'dart:io';

void main() {
  group('GPT2Attention', () {
    late Tensor hiddenStates;
    late Tensor attentionMask;
    late Tensor expectedOutput;
    late GPT2Attention attention;
    late Map<String, Tensor> tensors;

    setUpAll(() async {
      // Path to the generated safetensors file
      final path =
          '/home/tejag/projects/dart/ai/testdata/test_data/llm/gpt2/gpt2_attention.safetensors';
      final file = File(path);
      if (!file.existsSync()) {
        throw Exception(
          'Test data not found at $path. Run generation script first.',
        );
      }

      final loader = SafeTensorLoader.fromFile(file);
      tensors = await loader.load();

      // Load helper to simulate a "loader" for the module relative keys if needed
      // But loadFromSafeTensor takes the loader itself.

      // Parameters from generation script
      final int numHeads = 12;
      final int embedDim = 64 * numHeads;
      final int layerIdx = 0;
      final double dropout = 0.0;
      final int maxPositionEmbeddings = 1024;

      attention = await GPT2Attention.loadFromSafeTensor(
        loader,
        prefix: '', // Keys are at root: c_attn.weight, etc.
        name: 'test_attention',
        layerIdx: layerIdx,
        attentionDropoutProbability: dropout,
        residualDropoutProbability: dropout,
        isCrossAttention: false,
        numHeads: numHeads,
        scaleAttnByInverseLayerIdx: false,
        maxPositionEmbeddings: maxPositionEmbeddings,
        attnFuncType: GPT2AttentionMethodType.eager,
      );

      hiddenStates = tensors['hidden_states']!;
      attentionMask = tensors['attention_mask']!;
      expectedOutput = tensors['output']!;
    });

    test('forward pass matches pytorch', () {
      final context = Context.best();

      final result = attention.forward(
        hiddenStates,
        attentionMask: attentionMask,
        context: context,
      );

      final diff = (result.attentionOutput - expectedOutput)
          .abs()
          .max()
          .item<double>();
      print('Max difference: $diff');

      // The tolerance might need adjustment depending on float precision
      expect(diff, closeTo(0.0, 1e-4));
    });
  });
}
