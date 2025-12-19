import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

// Simple mock tokenizer for testing
class MockTokenizer {
  final Map<String, int> vocab;
  final Map<int, String> reverseVocab;

  MockTokenizer(List<String> words)
    : vocab = {for (var i = 0; i < words.length; i++) words[i]: i},
      reverseVocab = {for (var i = 0; i < words.length; i++) i: words[i]};

  Tensor encode(String text) {
    final tokens = text.split(' ').map((w) => vocab[w] ?? 0).toList();
    return Tensor.from(tokens, [1, tokens.length], dataType: DataType.int64);
  }

  String decode(Tensor tokens) {
    tokens = tokens.to(device: Device.cpu).flatten();
    final List<int> ids = [];
    for (int i = 0; i < tokens.shape[0]; i++) {
      ids.add(tokens.at([i]).scalar as int);
    }
    return ids.map((id) => reverseVocab[id] ?? '<unk>').join(' ');
  }

  int get vocabSize => vocab.length;
}

void main() {
  group('GPT2 Generation', () {
    late GPT2LMHeadModel model;
    late MockTokenizer tokenizer;
    late Context context;

    setUp(() {
      // Create a small vocabulary
      final words = [
        '<pad>',
        'Question:',
        'What',
        'is',
        '1+1?',
        'Answer:',
        '2',
        'The',
        'capital',
        'of',
        'France',
        'Paris',
        'dog',
        'cat',
      ];
      tokenizer = MockTokenizer(words);

      final config = GPT2Config(
        vocabSize: words.length,
        nPositions: 20,
        embedDim: 32,
        nLayer: 2,
        nHead: 4,
      );

      model = GPT2LMHeadModel.make(config: config, name: 'gpt2');
      context = Context.best();
    });

    test('Generate answer for 1+1', () {
      final prompt = "Question: What is 1+1? Answer:";
      final inputIds = tokenizer.encode(prompt);

      // Generate 1 new token
      final output = model.generate(
        inputIds,
        maxNewTokens: 1,
        context: context,
      );

      expect(output.shape, [1, inputIds.shape[1] + 1]);

      final generatedText = tokenizer.decode(output);
      print('Prompt: $prompt');
      print('Generated: $generatedText');

      // Since weights are random, we can't expect "2", but we check structure
      expect(generatedText.startsWith(prompt), isTrue);
    });

    test('Generate answer for capital of France', () {
      final prompt = "Question: What is The capital of France Answer:";
      final inputIds = tokenizer.encode(prompt);

      // Generate 1 new token
      final output = model.generate(
        inputIds,
        maxNewTokens: 1,
        context: context,
      );

      expect(output.shape, [1, inputIds.shape[1] + 1]);

      final generatedText = tokenizer.decode(output);
      print('Prompt: $prompt');
      print('Generated: $generatedText');

      expect(generatedText.startsWith(prompt), isTrue);
    });

    test('Generate multiple tokens', () {
      final prompt = "The dog";
      final inputIds = tokenizer.encode(prompt);

      // Generate 3 new tokens
      final output = model.generate(
        inputIds,
        maxNewTokens: 3,
        context: context,
      );

      expect(output.shape, [1, inputIds.shape[1] + 3]);

      final generatedText = tokenizer.decode(output);
      print('Prompt: $prompt');
      print('Generated: $generatedText');

      expect(generatedText.startsWith(prompt), isTrue);
      expect(generatedText.split(' ').length, prompt.split(' ').length + 3);
    });
  });
}
