import 'dart:io';
import 'package:tensor/tensor.dart';
import 'package:kamma/src/llama/llama_decoder_layer.dart';
import 'package:test/test.dart';

class TestCase {
  final String name;
  final int numHeads;
  final double ropeTheta;
  final int layerIdx;
  final int maxPositionEmbeddings;
  final int hiddenSize;
  final int numKeyValueHeads;
  final int headDim;
  final double rmsNormEps;
  final bool isCausal;

  final Tensor hiddenStates;
  final Tensor attentionMask;
  final Tensor positionIds;
  final Tensor cos;
  final Tensor sin;
  final Tensor expectedOutput;

  final Tensor inputLayernormOut;
  final Tensor selfAttnOut;
  final Tensor postAttentionLayernormOut;
  final Tensor mlpOut;

  TestCase({
    required this.name,
    required this.numHeads,
    required this.ropeTheta,
    required this.layerIdx,
    required this.maxPositionEmbeddings,
    required this.hiddenSize,
    required this.numKeyValueHeads,
    required this.headDim,
    required this.rmsNormEps,
    required this.isCausal,
    required this.hiddenStates,
    required this.attentionMask,
    required this.positionIds,
    required this.cos,
    required this.sin,
    required this.expectedOutput,
    required this.inputLayernormOut,
    required this.selfAttnOut,
    required this.postAttentionLayernormOut,
    required this.mlpOut,
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

    // Intermediate outputs
    final inputLayernormOut = await loader.loadByName(
      '$name.input_layernorm_out',
      device: device,
    );
    final selfAttnOut = await loader.loadByName(
      '$name.self_attn_out',
      device: device,
    );
    final postAttentionLayernormOut = await loader.loadByName(
      '$name.post_attention_layernorm_out',
      device: device,
    );
    final mlpOut = await loader.loadByName('$name.mlp_out', device: device);

    return TestCase(
      name: name,
      numHeads: int.parse(metadata['$name.num_heads']!),
      ropeTheta: double.parse(metadata['$name.rope_theta']!),
      layerIdx: int.parse(metadata['$name.layer_idx']!),
      maxPositionEmbeddings: int.parse(
        metadata['$name.max_position_embeddings']!,
      ),
      hiddenSize: int.parse(metadata['$name.hidden_size']!),
      numKeyValueHeads: int.parse(metadata['$name.num_kv_heads']!),
      headDim: int.parse(metadata['$name.head_dim']!),
      rmsNormEps: double.parse(metadata['$name.rms_norm_eps']!),
      isCausal: metadata['$name.is_causal']!.toLowerCase() == 'true',
      hiddenStates: hiddenStates,
      attentionMask: attentionMask,
      positionIds: positionIds,
      cos: cos,
      sin: sin,
      expectedOutput: expectedOutput,
      inputLayernormOut: inputLayernormOut,
      selfAttnOut: selfAttnOut,
      postAttentionLayernormOut: postAttentionLayernormOut,
      mlpOut: mlpOut,
    );
  }
}

void main() {
  group('LlamaDecoderLayer TinyLlama 1.1B', () {
    late SafeTensorsFile file;
    late SafeTensorLoader loader;

    setUpAll(() async {
      final path =
          '../testdata/test_data/llm/llama/tinyllama_1dot1/decoder_layer_test.safetensors';
      if (!File(path).existsSync()) {
        throw Exception(
          'Test data not found. Run python test generation script first. Path: $path',
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

        // Load DecoderLayer
        final decoderLayer = await LlamaDecoderLayer.loadFromSafeTensor(
          loader,
          prefix: '$name.',
          layerIdx: test.layerIdx,
          embedDim: test.hiddenSize,
          numHeads: test.numHeads,
          attentionDropoutProb: 0.0,
          maxPositionEmbeddings: test.maxPositionEmbeddings,
          ropeTheta: test.ropeTheta,
          isCausal: test.isCausal,
          rmsNormEps: test.rmsNormEps,
          activation: Activation.silu,
        );

        final context = Context(device: Device.cpu);

        // 1. Input Norm
        final inputNormOut = decoderLayer.inputLayernorm.forward(
          test.hiddenStates,
          context: context,
        );
        expect(
          inputNormOut.allClose(test.inputLayernormOut, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: Input Norm mismatch',
        );

        // 2. Self Attention
        final attnOut = decoderLayer.selfAttn.forward(
          inputNormOut,
          context: context,
          attentionMask: test.attentionMask,
          positionIds: test.positionIds,
          positionEmbeddings: (cos: test.cos, sin: test.sin),
        );
        expect(
          attnOut.allClose(test.selfAttnOut, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: Self Attention mismatch',
        );

        // 3. Post Attention Norm
        // Residual connection handled inside DecoderLayer?
        // Test logic manually verifies components step-by-step to be sure.
        // But here we want to verify DecoderLayer behavior OR components?
        // Since we verify components above, we can assume they are correct.
        // But let's verify Post Attention Norm input correctness (Residual).
        final residual1 = test.hiddenStates + attnOut;

        final postAttnNormOut = decoderLayer.postAttentionLayernorm.forward(
          residual1,
          context: context,
        );
        expect(
          postAttnNormOut.allClose(
            test.postAttentionLayernormOut,
            atol: 1e-4,
            rtol: 1e-4,
          ),
          isTrue,
          reason: '$name: Post Attention Norm mismatch',
        );

        // 4. MLP
        final mlpOut = decoderLayer.mlp.forward(
          postAttnNormOut,
          context: context,
        );
        expect(
          mlpOut.allClose(test.mlpOut, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: MLP mismatch',
        );

        // Final Output (End-to-End check)
        final output = decoderLayer.forward(
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
        expect(
          close,
          isTrue,
          reason: '$name: Output does not match PyTorch output',
        );
      }
    });
  });
}
