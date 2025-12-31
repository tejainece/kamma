import 'dart:convert';
import 'dart:io';

import 'package:kamma/kamma.dart';
import 'package:test/test.dart';
import 'package:path/path.dart' as path;

void main() {
  group('GPT2 Inference', () {
    final modelDir = '/Users/tejag/projects/dart/ai/testdata/models/llm/gpt2';
    final testDataPath =
        '/Users/tejag/projects/dart/ai/testdata/test_data/llm/gpt2/inference/model/test.json';
    Context context = Context.best();

    late GPT2LMHeadModel model;
    late GPT2Tokenizer tokenizer;
    late List<Map<String, dynamic>> testData;

    setUpAll(() async {
      final configFile = File(path.join(modelDir, 'config.json'));
      final configJson = json.decode(await configFile.readAsString());
      final config = GPT2Config.fromJson(configJson);

      tokenizer = await GPT2Tokenizer.fromPretrained(modelDir);

      final safeTensorFile = await SafeTensorsFile.load(
        path.join(modelDir, 'model.safetensors'),
      );
      final loader = safeTensorFile.mmapTensorLoader();

      print('Loading model...');
      model = await GPT2LMHeadModel.loadFromSafeTensor(loader, config: config);
      print('Model loaded.');

      final jsonContent = await File(testDataPath).readAsString();
      final List<dynamic> jsonList = json.decode(jsonContent);
      testData = jsonList.cast<Map<String, dynamic>>();
      print('Test data loaded.');
    });

    test('Verification against HuggingFace implementation', () {
      print('Starting verification test...');
      for (final testCase in testData) {
        // TODO perform this inside generate, etc
        model.resetKeyValueCache();

        final prompt = testCase['prompt'] as String;
        final expectedResponse = testCase['response'] as String;

        print('Testing prompt: "$prompt"');

        final inputIds = tokenizer.encode(prompt);
        print('Input encoded: shape=${inputIds.shape}');

        bool logitsMatch = false;
        if (testCase.containsKey('first_token_logits')) {
          print('Checking logits for prompt: "$prompt"');
          // Forward pass only, no generation
          final logits = model.forward(inputIds, context: context);
          // Logits shape: [batch, seq_len, vocab_size]
          // We want logits of last token: [0, -1, :]
          final lastTokenLogits = logits
              .select(1, logits.shape[1] - 1)
              .select(0, 0);  
          final expectedLogits = (testCase['first_token_logits'] as List)
              .cast<double>();

          print('Logits shape: ${lastTokenLogits.shape}');
          print('Expected logits len: ${expectedLogits.length}');

          // Compare a few values
          int mismatchedLogits = 0;
          for (int i = 0; i < 10; i++) {
            final actual = lastTokenLogits[i].scalar as double;
            final expected = expectedLogits[i];
            print(
              'Logit[$i]: Example (Actual: $actual, Expected: $expected, Diff: ${(actual - expected).abs()})',
            );

            if ((actual - expected).abs() > 1e-3) {
              mismatchedLogits++;
              print("MISMATCH at index $i");
            }
          }
          // Reset again after forward pass just to be safe before generation,
          // though generating starts fresh usually (but generate internal loop uses cache).
          // Actually model.generate usually handles its own loop, but if model state is persistent...
          // model.generate calls forward(). The forward() updates the cache.
          // So we MUST reset after the logits check too because that forward() populated the cache!
          model.resetKeyValueCache();

          if (mismatchedLogits == 0) {
            logitsMatch = true;
          } else {
            fail("Logits mismatched significantly.");
          }
        }

        // Generate
        // We use temperature 0.0 for greedy decoding to match the deterministic test data generation
        print('Generating...');
        final outputIds = model.generate(
          inputIds,
          maxNewTokens: 20,
          temperature: 0.0,
          context: context,
        );
        print('Generated outputIds: shape=${outputIds.shape}');

        // Decode output
        final generatedText = tokenizer.decode(outputIds);
        print('Decoded text: "$generatedText"');

        // Verify
        // Note: decode might include the input prompt depending on how generate returns
        // GPT2LMHeadModel.generate returns "currentInputIds" which includes the original input.

        // Use strict check now that state leak is fixed.
        // We still keep the warning logic if logits matched but text differs, just in case of pure precision causing 1 token flip,
        // but hopefully it matches now.
        if (logitsMatch && generatedText != expectedResponse) {
          print(
            "WARNING: Generated text differs from expected, but logits matched. Accepting due to float precision divergence.",
          );
          print("Expected: $expectedResponse");
          print("Actual:   $generatedText");
        }
        expect(generatedText, equals(expectedResponse));
      }
    });
  });
}
