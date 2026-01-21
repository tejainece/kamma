import 'dart:io';
import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

void main() async {
  final modelDir = '/Users/tejag/projects/dart/ai/testdata/models/llm/gpt2';

  // Check if model exists, otherwise skip (to avoid CI failures if data missing)
  if (!Directory(modelDir).existsSync()) {
    return;
  }

  /*
  // We can't easily test full generation without loading the big model, which might be slow.
  // But we can test that the method accepts the arguments.
  // For a real functional test, we'd need to load the model.
  // Let's try to load it, similar to inference test.
  */

  group('GPT2 Prompt', () {
    late GPT2 gpt2;
    late Context context;

    setUpAll(() async {
      context = Context.best();
      gpt2 = await GPT2.loadFromDirectory(modelDir);
    });

    test('prompt accepts parameters', () {
      // This test mainly verifies that the code compiles and runs with these named arguments.
      // We use a very short maxNewTokens to be quick.
      final result = gpt2.prompt(
        "Hello world",
        context: context,
        maxNewTokens: 2,
        temperature: 0.7,
        topK: 50,
        topP: 0.9,
      );
      print('Generated: $result');
      expect(result, isNotEmpty);
      expect(result, isA<String>());
    });

    test('prompt works with defaults', () {
      final result = gpt2.prompt("Hello world", context: context);
      expect(result, isNotEmpty);
    });
  });
}
