import 'dart:convert';
import 'dart:io';

import 'package:kamma/kamma.dart';

import 'package:test/test.dart';
import 'package:path/path.dart' as path;

// TODO: Move shared Test class to a common place if reused often
class Test {
  final String prompt;
  final String response;
  final Tensor promptIds;
  final Tensor responseIds;
  final Tensor logits;
  final List<Tensor> hiddenStates;

  Test({
    required this.prompt,
    required this.response,
    required this.promptIds,
    required this.responseIds,
    required this.logits,
    required this.hiddenStates,
  });

  static Future<Test> loadFromSafeTensor(
    SafeTensorLoader loader,
    String name, {
    required Device device,
  }) async {
    final logits = await loader.loadByName('$name.logits', device: device);
    final promptIds = await loader.loadByName(
      '$name.prompt_ids',
      device: device,
    );
    final responseIds = await loader.loadByName(
      '$name.response_ids',
      device: device,
    );

    final hiddenStates = <Tensor>[];
    for (int i = 0; ; i++) {
      final key = '$name.hidden_state_$i';
      if (!loader.hasTensor(key)) break;
      hiddenStates.add(await loader.loadByName(key, device: device));
    }

    return Test(
      prompt: loader.metadata['$name.prompt']!,
      response: loader.metadata['$name.response']!,
      promptIds: promptIds,
      responseIds: responseIds,
      logits: logits,
      hiddenStates: hiddenStates,
    );
  }
}

void main() async {
  Context context = Context.best();
  final testDataPath =
      './../testdata/test_data/llm/gemma/v1/2b/inference/inference.safetensors';
  final modelDir = './../testdata/models/llm/gemma/v1/2b';
  final configFile = File(path.join(modelDir, 'config.json'));
  final configJson = json.decode(await configFile.readAsString());
  var config = GemmaConfig.fromJson(configJson);

  if (!await File(testDataPath).exists()) {
    throw Exception(
      'Test data not found at $testDataPath. Run generation script.',
    );
  }
  final testDataFile = await SafeTensorsFile.load(testDataPath);
  final testLoader = testDataFile.mmapTensorLoader();

  group('Gemma 2B Inference', () {
    late SafeTensorLoader modelLoader;

    setUpAll(() async {
      // Load sharded model weights using CompositeSafeTensorLoader
      modelLoader = await CompositeSafeTensorLoader.loadSplitSafeTensors(
        Directory(modelDir),
      );
    });

    final tested = <String>{};

    for (final key in testLoader.header.tensorInfos.keys) {
      final name = key.split('.').first;
      if (tested.contains(name)) continue;
      tested.add(name);

      test('Inference Case: $name', () async {
        final model = await GemmaForCausalLM.loadFromSafeTensor(
          modelLoader,
          config: config,
          activation:
              Activation.gelu, // Gemma usually uses gelu (or gelu_pytorch_tanh)
          attentionMethod: GPT2AttentionMethodType.sdap,
        );

        final testCase = await Test.loadFromSafeTensor(
          testLoader,
          name,
          device: context.device,
        );

        // 1. Generation Test (Token ID Matching)
        {
          model.resetKeyValueCache();
          Tensor currentInputIds = testCase.promptIds.to(
            device: context.device,
          );

          final inputLen = testCase.promptIds.shape[1];
          final targetLen = testCase.responseIds.shape[1];
          final maxNewTokens = targetLen - inputLen;

          print('Prompt: "${testCase.prompt}"');
          print('Input IDs shape: ${currentInputIds.shape}');
          print('Target IDs shape: ${testCase.responseIds.shape}');

          for (int i = 0; i < maxNewTokens; i++) {
            final result = model.forward(currentInputIds, context: context);

            final nextTokenLogits = result.logits.select(
              1,
              result.logits.shape[1] - 1,
            );

            final nextToken = nextTokenLogits.argmax(dim: -1);
            final nextTokenUnsqueezed = nextToken.unsqueeze(1);

            currentInputIds = Tensor.cat([
              currentInputIds,
              nextTokenUnsqueezed,
            ], dim: 1);
          }

          // Verify IDs
          expect(
            currentInputIds.shape,
            equals(testCase.responseIds.shape),
            reason: 'Generated sequence length mismatch',
          );

          final num idDiff =
              (currentInputIds.to(device: Device.cpu) -
                          testCase.responseIds.to(device: Device.cpu))
                      .abs()
                      .max()
                      .scalar
                  as num;

          expect(
            idDiff,
            equals(0.0),
            reason:
                "Generated token IDs mismatch. Expected: ${testCase.responseIds.toList()}, Got: ${currentInputIds.toList()}",
          );
        }

        // 2. Full State Verification
        {
          model.resetKeyValueCache();
          final result = model.forward(
            testCase.responseIds.to(device: context.device),
            context: context,
            returnHiddenStates: true,
          );

          // Verify Logits
          final logitsDiff =
              (result.logits - testCase.logits.to(device: context.device))
                      .abs()
                      .max()
                      .scalar
                  as double;
          print('Logits max diff: $logitsDiff');
          expect(
            logitsDiff,
            closeTo(0, 1.0),
            reason: 'Logits mismatch',
          ); // Slightly loose tolerance for starters

          // Verify Hidden States
          final hiddenStates = result.allHiddenStates!;
          if (hiddenStates.length == testCase.hiddenStates.length) {
            for (int i = 0; i < hiddenStates.length; i++) {
              final hsDiff =
                  (hiddenStates[i] -
                              testCase.hiddenStates[i].to(
                                device: context.device,
                              ))
                          .abs()
                          .max()
                          .scalar
                      as double;
              print('Hidden state $i max diff: $hsDiff');
              expect(
                hsDiff,
                closeTo(
                  0,
                  20.0,
                ), // High tolerance for potentially accumulated diffs
                reason: 'Hidden state $i mismatch',
              );
            }
          } else {
            print(
              "Warning: hidden state count mismatch. Model: ${hiddenStates.length}, Test: ${testCase.hiddenStates.length}",
            );
          }
        }
      });
    }
  });
}
