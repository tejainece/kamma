import 'dart:convert';
import 'dart:io';
import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

void main() {
  group('GPT2Tokenizer', () {
    test('encode', () async {
      final tokenizer = await GPT2Tokenizer.fromPretrained('models/llm/gpt2');
      final text = 'Hello world';
      final tokens = tokenizer.encode(text);
      expect(tokens.toList(), [15496, 995]);
    });
    test('encode.withSpace', () async {
      final tokenizer = await GPT2Tokenizer.fromPretrained('models/llm/gpt2');
      final text = ' Hello world';
      final tokens = tokenizer.encode(text);
      expect(tokens.toList(), [18435, 995]);
    });
  });

  group('GPT2Tokenizer.dummy', () {
    late Directory tempDir;
    late String vocabPath;
    late String mergesPath;

    setUp(() async {
      tempDir = await Directory.systemTemp.createTemp('gpt2_tokenizer_test');
      vocabPath = '${tempDir.path}/vocab.json';
      mergesPath = '${tempDir.path}/merges.txt';

      // Create dummy vocab
      // Base vocab (bytes) + merged tokens
      final vocab = <String, int>{};

      // Add byte tokens (just a subset for testing)
      final byteEncoder = GPT2Tokenizer.bytesToUnicode();
      byteEncoder.forEach((b, u) {
        vocab[u] = b; // ID = byte value for simplicity in this test
      });

      String toToken(String s) {
        return utf8.encode(s).map((b) => byteEncoder[b]!).join('');
      }

      // Add some merged tokens
      // Let's say we merge 't' and 'h' -> 'th'
      // 'th' and 'e' -> 'the'
      // IDs for merged tokens start after 255
      vocab[toToken('th')] = 256;
      vocab[toToken('the')] = 257;
      vocab[toToken(' ')] = 258; // space
      vocab[toToken(' t')] = 259;
      vocab[toToken(' th')] = 260;
      vocab[toToken(' the')] = 261;

      await File(vocabPath).writeAsString(json.encode(vocab));

      // Create dummy merges
      // t h
      // th e
      // Ġ t
      // Ġt h
      // Ġth e
      final merges =
          '''
# version: 0.2
${toToken(' ')} t
${toToken(' t')} h
${toToken(' th')} e
t h
th e
''';
      await File(mergesPath).writeAsString(merges);
    });

    tearDown(() async {
      await tempDir.delete(recursive: true);
    });

    test('fromPretrained loads vocab and merges', () async {
      final tokenizer = await GPT2Tokenizer.fromPretrained(tempDir.path);
      expect(tokenizer.encoder['th'], 256);
      expect(tokenizer.bpeRanks['t h'], 3);
    });

    test('encode simple text', () async {
      final tokenizer = await GPT2Tokenizer.fromPretrained(tempDir.path);
      // 'the' should be token 257
      // 'the' -> 't' 'h' 'e' -> 'th' 'e' -> 'the'
      final text = 'the';
      final tokens = tokenizer.encode(text);
      expect(tokens.shape, [1, 1]);
      expect(tokens.at([0]).scalar, 257);
    });

    test('decode simple tokens', () async {
      final tokenizer = await GPT2Tokenizer.fromPretrained(tempDir.path);
      final tokens = Tensor.from([257], [1, 1], dataType: DataType.int64);
      final text = tokenizer.decode(tokens);
      expect(text, 'the');
    });

    test('encode with spaces', () async {
      final tokenizer = await GPT2Tokenizer.fromPretrained(tempDir.path);
      // ' the' -> 'Ġthe' (261)
      final text = ' the';
      final tokens = tokenizer.encode(text);
      expect(tokens.shape, [1, 1]);
      expect(tokens.at([0]).scalar, 261);
    });

    test('encode unknown chars (byte fallback)', () async {
      final tokenizer = await GPT2Tokenizer.fromPretrained(tempDir.path);
      // 'z' is not in our manual vocab merges, but it is in the byte vocab
      // So it should be encoded as its byte value
      final text = 'z';
      final tokens = tokenizer.encode(text);
      expect(tokens.shape, [1, 1]);
      // 'z' ascii is 122
      // In our setup, we mapped byte chars to their byte values if < 256
      // But wait, bytesToUnicode maps bytes to unicode chars.
      // And our vocab maps those unicode chars to IDs.
      // In setUp, we did: vocab[u] = b;
      // 'z' is byte 122. bytesToUnicode[122] = 'z'. vocab['z'] = 122.
      expect(tokens.at([0]).scalar, 122);
    });
  });
}
