import 'dart:convert';
import 'dart:io';

import 'package:kamma/kamma.dart';
import 'package:test/test.dart';
import 'package:path/path.dart' as path;

class Test {
  final String prompt;
  final String response;
  final Tensor promptIds;
  final Tensor inputIds;
  final Tensor logits;
  final List<Tensor> hiddenStates;
  final List<Tensor> attentions;

  Test({
    required this.prompt,
    required this.response,
    required this.promptIds,
    required this.inputIds,
    required this.logits,
    required this.hiddenStates,
    required this.attentions,
  });

  static Future<Test> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final promptIds = await loader.loadByName(
      '$name.prompt_ids',
      device: device,
    );
    final inputIds = await loader.loadByName('$name.input_ids', device: device);
    final logits = await loader.loadByName('$name.logits', device: device);

    final hiddenStates = <Tensor>[];
    for (int i = 0; ; i++) {
      final key = '$name.hidden_state_$i';
      if (!loader.hasTensor(key)) break;
      hiddenStates.add(await loader.loadByName(key, device: device));
    }

    final attentions = <Tensor>[];
    for (int i = 0; ; i++) {
      final key = '$name.attention_$i';
      if (!loader.hasTensor(key)) break;
      attentions.add(await loader.loadByName(key, device: device));
    }

    return Test(
      prompt: loader.metadata['$name.prompt']!,
      response: loader.metadata['$name.response']!,
      promptIds: promptIds,
      inputIds: inputIds,
      logits: logits,
      hiddenStates: hiddenStates,
      attentions: attentions,
    );
  }
}

void main() async {
  // Force CPU to avoid MPS crash with Causal Mask
  Context context = Context(device: Device.cpu);
  print('Running tests on device: ${context.device}');

  final testDataPath =
      '../testdata/test_data/llm/llama3dot2/1b/inference/llama3dot2_inference.safetensors';

  final modelDir = '../testdata/models/llm/llama/3dot2/1b';

  final configFile = File(path.join(modelDir, 'config.json'));
  final configJson = json.decode(await configFile.readAsString());
  var config = LlamaConfig.fromJson(configJson);

  final modelFile = await SafeTensorsFile.load(
    path.join(modelDir, 'model.safetensors'),
  );
  final modelLoader = modelFile.mmapTensorLoader();

  group('Llama 3.2 1B Inference', () {
    late SafeTensorsFile testSafeTensorFile;
    late MmapSafeTensorLoader testDataLoader;

    setUpAll(() async {
      testSafeTensorFile = await SafeTensorsFile.load(testDataPath);
      testDataLoader = testSafeTensorFile.mmapTensorLoader();
    });

    test('Verify Generation and Logits', () async {
      final model = await LlamaForCausalLM.loadFromSafeTensor(
        modelLoader,
        config,
      );

      final tested = <String>{};

      for (final key in testDataLoader.header.tensorInfos.keys) {
        final name = key.split('.').first;
        if (!name.startsWith('test')) continue;
        if (tested.contains(name)) continue;
        tested.add(name);

        print('Testing case: $name');

        final testCase = await Test.loadFromSafeTensor(
          testDataLoader,
          name,
          device: context.device,
        );

        // --- Verify Hidden States ---
        final fullInputIds = testCase.inputIds.to(device: context.device);
        final result = model.forward(
          fullInputIds,
          context: context,
          returnHiddenStates: true,
        );
        final logits = result.logits;
        final hiddenStates = result.hiddenStates!;
        final expectedLogits = testCase.logits.to(device: context.device);

        // Verify layer outputs
        // Note: result.hiddenStates[0] is embeddings output
        print('  Embeddings (Layer -1):');
        final embeddings = hiddenStates[0];
        print(
          '    Mean: ${embeddings.mean().scalar}, Min: ${embeddings.min().scalar}, Max: ${embeddings.max().scalar}',
        );

        print('  Verifying ${testCase.hiddenStates.length} hidden states...');
        for (int i = 0; i < testCase.hiddenStates.length; i++) {
          // hiddenStates[0] is Embeddings.
          // testCase.hiddenStates[0] is Embeddings (HF standard).
          final actual = hiddenStates[i];
          final expected = testCase.hiddenStates[i].to(device: context.device);

          final diff = (actual - expected).abs().max().scalar as double;
          print('    Layer $i max diff: $diff');
          print(
            '    ACTUAL: Mean: ${actual.mean().scalar}, Min: ${actual.min().scalar}, Max: ${actual.max().scalar}',
          );
          print(
            '    EXPECT: Mean: ${expected.mean().scalar}, Min: ${expected.min().scalar}, Max: ${expected.max().scalar}',
          );

          /*
          expect(
            diff,
            closeTo(0, 1e-3),
            reason: 'Hidden state mismatch at layer $i',
          );
          */
        }

        final diff = (logits - expectedLogits).abs().max().scalar as double;
        print('  Logits max diff: $diff');

        expect(
          diff,
          closeTo(0, 1e-2),
          reason: 'Logits mismatch for case $name',
        );

        // --- Verify Generation ---
        var currentIds = testCase.promptIds.to(device: context.device);
        final expectedIds = testCase.inputIds.to(
          device: context.device,
        ); // Full sequence

        final promptLen = currentIds.shape[1];
        final totalLen = expectedIds.shape[1];
        final maxNewTokens = totalLen - promptLen;

        print('  Generating $maxNewTokens tokens...');

        for (int i = 0; i < maxNewTokens; i++) {
          // Create attention mask if needed (assuming left padding or no padding for simple generation)
          // For greedy generation with batch size 1, no padding mask needed usually if implicit.

          final result = model.forward(currentIds, context: context);
          final logits = result.logits;
          // Get logits for the last token to predict next
          // logits shape: [batch, seq_len, vocab_size]
          // We want the last step.
          final lastLogits = logits.slice(1, -1); // [batch, 1, vocab]

          final nextToken = lastLogits.argmax(dim: -1); // [batch, 1]

          final actualToken = nextToken.scalar as int;
          final expectedToken =
              expectedIds.at([0, promptLen + i]).scalar as int;

          expect(
            actualToken,
            equals(expectedToken),
            reason: 'Token mismatch at step $i',
          );

          currentIds = Tensor.cat([currentIds, nextToken], dim: 1);
        }
        print('  Generation passed.');
      }
    });
  });
}
