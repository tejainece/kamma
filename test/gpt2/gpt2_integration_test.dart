import 'dart:convert';
import 'dart:io';
import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

void main() {
  group('GPT2 Integration', () {
    late Directory tempDir;
    late String vocabPath;
    late String mergesPath;
    late GPT2Tokenizer tokenizer;
    late GPT2LMHeadModel model;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('gpt2_integration_test');
      vocabPath = '${tempDir.path}/vocab.json';
      mergesPath = '${tempDir.path}/merges.txt';

      // 1. Setup Tokenizer (same as gpt2_tokenizer_test.dart)
      final vocab = <String, int>{};
      final byteEncoder = GPT2Tokenizer.bytesToUnicode();
      byteEncoder.forEach((b, u) {
        vocab[u] = b;
      });

      String toToken(String s) {
        return utf8.encode(s).map((b) => byteEncoder[b]!).join('');
      }

      // Add tokens needed for "Question: What is 1+1? Answer:"
      // We need to ensure all characters are covered.
      // The byte encoder covers all bytes, so basic chars are covered.
      // We can add some merges to make it more realistic, but not strictly necessary for mechanics.
      // Let's add 'Question', 'What', 'is', 'Answer' as tokens for cleaner output if we wanted,
      // but for now, byte-level fallback is fine.
      // Actually, let's add the specific words to vocab to ensure they are single tokens if possible,
      // or just rely on the fact that they will be broken down.
      // Let's just use the basic byte vocab + some common merges.

      vocab[toToken('Question')] = 300;
      vocab[toToken('What')] = 301;
      vocab[toToken('is')] = 302;
      vocab[toToken('Answer')] = 303;
      vocab[toToken(' ')] = 304;
      vocab[toToken(':')] = 305;
      vocab[toToken('?')] = 306;
      vocab[toToken('1')] = 307;
      vocab[toToken('+')] = 308;

      await File(vocabPath).writeAsString(json.encode(vocab));

      final merges = '''
# version: 0.2
''';
      await File(mergesPath).writeAsString(merges);

      tokenizer = await GPT2Tokenizer.fromPretrained(tempDir.path);

      // 2. Setup Model
      // Use a small config for speed
      final config = GPT2Config(
        vocabSize: 1000, // Make sure this is larger than our max vocab ID
        embedDim: 32,
        numLayers: 2,
        numHeads: 2,
        nPositions: 128,
      );
      model = GPT2LMHeadModel.make(config: config, name: 'gpt2');
    });

    tearDown(() async {
      await tempDir.delete(recursive: true);
    });

    test('Full pipeline generation', () async {
      final prompt = "Question: What is 1+1? Answer:";
      print('Prompt: $prompt');

      // Check for downloaded weights
      final modelPath = 'model/llm/gpt';
      final safetensorsFile = File('$modelPath/model.safetensors');

      if (await safetensorsFile.exists()) {
        print('Loading weights from $modelPath');
        // Re-initialize tokenizer and model with downloaded files
        tokenizer = await GPT2Tokenizer.fromPretrained(modelPath);

        // Load config from file if possible, but for now we use hardcoded config matching GPT-2 small
        final config = GPT2Config(
          vocabSize: 50257,
          embedDim: 768,
          numLayers: 12,
          numHeads: 12,
          nPositions: 1024,
        );
        // model = GPT2LMHeadModel.make(config: config, name: 'gpt2');
        final safetensors = await SafeTensorsFile.load(safetensorsFile.path);
        final loader = safetensors.mmapTensorLoader();
        model = await GPT2LMHeadModel.loadFromSafeTensor(
          loader,
          prefix: '',
          config: config,
        );
      } else {
        print('Weights not found, using random weights and dummy vocab');
      }

      // 1. Encode
      final inputTokens = tokenizer.encode(prompt);
      print('Input tokens shape: ${inputTokens.shape}');

      // 2. Generate
      final context = Context.best();
      // Move model to device
      // Note: We don't have a model.to(device) yet, but weights are created on Context.best()
      // if we ensure context is set correctly or if we just rely on default.
      // The model weights are created using Tensor.empty/randn which might default to CPU or CUDA depending on implementation.
      // In gpt2_generation_test we saw weights on CUDA.
      // Let's assume model is on the same device as inputTokens will be moved to.

      final outputTokens = model.generate(
        inputTokens,
        maxNewTokens: 10, // Generate more tokens to see if it makes sense
        context: context,
      );
      print('Output tokens shape: ${outputTokens.shape}');

      // 3. Decode
      final outputText = tokenizer.decode(outputTokens);
      print('Generated text: $outputText');

      // 4. Verify
      expect(outputText, startsWith(prompt));
      expect(outputText, startsWith(prompt));
      // With random weights, we might generate tokens not in our tiny vocab, so outputText might not grow.
      // But we confirmed outputTokens shape grew.
      expect(outputTokens.shape[1], greaterThan(inputTokens.shape[1]));
    });
  });
}
