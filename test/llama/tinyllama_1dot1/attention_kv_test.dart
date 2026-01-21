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

  // Prefill Data
  final Tensor prefillInput;
  final Tensor prefillAttentionMask;
  final Tensor prefillPositionIds;
  final Tensor prefillCos;
  final Tensor prefillSin;
  final Tensor prefillOutput;
  final Tensor prefillPastKey;
  final Tensor prefillPastValue;

  // Decode Data
  final Tensor decodeHiddenStates;
  final Tensor decodeAttentionMask;
  final Tensor decodePositionIds;
  final Tensor decodeCos;
  final Tensor decodeSin;
  final Tensor decodeOutput;
  final Tensor decodePastKey;
  final Tensor decodePastValue;

  TestCase({
    required this.name,
    required this.numHeads,
    required this.ropeTheta,
    required this.layerIdx,
    required this.maxPositionEmbeddings,
    required this.prefillInput,
    required this.prefillAttentionMask,
    required this.prefillPositionIds,
    required this.prefillCos,
    required this.prefillSin,
    required this.prefillOutput,
    required this.prefillPastKey,
    required this.prefillPastValue,
    required this.decodeHiddenStates,
    required this.decodeAttentionMask,
    required this.decodePositionIds,
    required this.decodeCos,
    required this.decodeSin,
    required this.decodeOutput,
    required this.decodePastKey,
    required this.decodePastValue,
  });

  static Future<TestCase> load(SafeTensorLoader loader, String name) async {
    final device = Device.cpu;
    final metadata = loader.metadata;

    // Helper to load
    Future<Tensor> loadT(String suffix) async =>
        loader.loadByName('$name.$suffix', device: device);

    return TestCase(
      name: name,
      numHeads: int.parse(metadata['$name.num_heads']!),
      ropeTheta: double.parse(metadata['$name.rope_theta']!),
      layerIdx: int.parse(metadata['$name.layer_idx']!),
      maxPositionEmbeddings: int.parse(
        metadata['$name.max_position_embeddings']!,
      ),

      prefillInput: await loadT('prefill.hidden_states'),
      prefillAttentionMask: await loadT('prefill.attention_mask'),
      prefillPositionIds: await loadT('prefill.position_ids'),
      prefillCos: await loadT('prefill.cos'),
      prefillSin: await loadT('prefill.sin'),
      prefillOutput: await loadT('prefill.output'),
      prefillPastKey: await loadT('prefill.past_key'),
      prefillPastValue: await loadT('prefill.past_value'),

      decodeHiddenStates: await loadT('decode.hidden_states'),
      decodeAttentionMask: await loadT('decode.attention_mask'),
      decodePositionIds: await loadT('decode.position_ids'),
      decodeCos: await loadT('decode.cos'),
      decodeSin: await loadT('decode.sin'),
      decodeOutput: await loadT('decode.output'),
      decodePastKey: await loadT('decode.past_key'),
      decodePastValue: await loadT('decode.past_value'),
    );
  }
}

void main() {
  group('LlamaAttention KV Cache TinyLlama 1.1B', () {
    late SafeTensorsFile file;
    late SafeTensorLoader loader;

    setUpAll(() async {
      final path =
          './testdata/test_data/llm/llama/tinyllama_1dot1/attention_kv_test.safetensors';
      if (!File(path).existsSync()) {
        throw Exception(
          'Test data not found at $path. Run python test generation script first.',
        );
      }
      file = await SafeTensorsFile.load(path);
      loader = file.mmapTensorLoader();
    });

    test('prefill and decode steps match PyTorch', () async {
      final seen = <String>{};
      for (final key in loader.tensorInfos.keys) {
        final name = key.split('.').first;
        if (seen.contains(name)) continue;
        seen.add(name);

        print('Running test for $name');
        final test = await TestCase.load(loader, name);

        // Load attention module
        // We assume weights are named "self_attn.q_proj.weight" etc inside the file?
        // Wait, the python script saved state_dict as "tc.name.self_attn.k" etc? No.
        // The python script does NOT save weights in my current version!
        // Oops, I forgot to save the weights in the python script.
        // It saves "tensors" but I didn't add "attn.state_dict()" to "tensors".
        // I need to fix the python script to save weights, otherwise I can't load the module.

        // Assuming weights are saved (I will fix python script momentarily):
        final attention = await LlamaAttention.loadFromSafeTensor(
          loader,
          prefix: '$name.self_attn.', // Expecting weights here
          name: 'self_attn',
          layerIdx: test.layerIdx,
          numHeads: test.numHeads,
          maxPositionEmbeddings: test.maxPositionEmbeddings,
          ropeTheta: test.ropeTheta,
          attentionDropoutProb: 0.0,
          isCausal: true,
        );

        final context = Context(device: Device.cpu);

        // --- Prefill Step ---
        print('  Prefill step...');
        final outputPrefill = attention.forward(
          test.prefillInput,
          context: context,
          attentionMask: test.prefillAttentionMask,
          positionIds: test.prefillPositionIds,
          positionEmbeddings: (cos: test.prefillCos, sin: test.prefillSin),
          useCache: true,
        );

        // Check Output
        expect(
          outputPrefill.allClose(test.prefillOutput, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: Prefill output mismatch',
        );

        // Check KV Cache
        // Cache API: attention.keyValueCache.key, attention.keyValueCache.value
        final cacheKey = attention.keyValueCache.key;
        final cacheValue = attention.keyValueCache.value;

        expect(
          cacheKey.allClose(test.prefillPastKey, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: Prefill KV Cache Key mismatch',
        );
        expect(
          cacheValue.allClose(test.prefillPastValue, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: Prefill KV Cache Value mismatch',
        );

        // --- Decode Step ---
        print('  Decode step...');
        final outputDecode = attention.forward(
          test.decodeHiddenStates,
          context: context,
          attentionMask: test.decodeAttentionMask,
          positionIds: test.decodePositionIds,
          positionEmbeddings: (cos: test.decodeCos, sin: test.decodeSin),
          useCache: true,
        );

        // Check Output
        expect(
          outputDecode.allClose(test.decodeOutput, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: Decode output mismatch',
        );

        // Check Updated KV Cache
        final cacheKeyUpdated = attention.keyValueCache.key;
        final cacheValueUpdated = attention.keyValueCache.value;

        expect(
          cacheKeyUpdated.allClose(test.decodePastKey, atol: 1e-4, rtol: 1e-4),
          isTrue,
          reason: '$name: Decode KV Cache Key mismatch',
        );
        expect(
          cacheValueUpdated.allClose(
            test.decodePastValue,
            atol: 1e-4,
            rtol: 1e-4,
          ),
          isTrue,
          reason: '$name: Decode KV Cache Value mismatch',
        );
      }
    });
  });
}
