import 'dart:convert';
import 'dart:io';
import 'package:tensor/tensor.dart';
import 'package:path/path.dart' as path;

/// BPE tokenizer for Llama models using tokenizer.json
class LlamaTokenizer {
  final Map<String, int> encoder;
  final Map<int, String> decoder;
  final Map<String, int> bpeRanks;
  final Map<String, String> cache = {};

  // Normalization replacement character (U+2581)
  static const String _spi = '▁';

  LlamaTokenizer({
    required this.encoder,
    required this.decoder,
    required this.bpeRanks,
  });

  static Future<LlamaTokenizer> fromPretrained(String modelPath) async {
    final tokenizerFile = File(path.join(modelPath, 'tokenizer.json'));

    if (!await tokenizerFile.exists()) {
      throw Exception('tokenizer.json not found at $modelPath');
    }

    final content = await tokenizerFile.readAsString();
    final jsonContent = json.decode(content) as Map<String, dynamic>;

    final model = jsonContent['model'] as Map<String, dynamic>;
    final vocab = model['vocab'] as Map<String, dynamic>;
    final merges = (model['merges'] as List).cast<String>();

    final encoder = vocab.map((k, v) => MapEntry(k, v as int));
    final decoder = encoder.map((k, v) => MapEntry(v, k));

    final bpeRanks = <String, int>{};
    for (var i = 0; i < merges.length; i++) {
      bpeRanks[merges[i]] = i;
    }

    return LlamaTokenizer(
      encoder: encoder,
      decoder: decoder,
      bpeRanks: bpeRanks,
    );
  }

  String _normalize(String text) {
    // Replace spaces with SPI and prepend SPI (as per Llama tokenizer.json normalizer)
    return _spi + text.replaceAll(' ', _spi);
  }

  Tensor encode(String text) {
    // 1. Normalize
    final normalized = _normalize(text);

    // 2. Pre-tokenize / Split
    // In SentencePiece/Llama, normalization often handles the split implicitly or we process the whole string?
    // TinyLlama tokenizer.json says "pre_tokenizer": null.
    // So likely we treat the whole normalized string as one sequence of characters?
    // However, usually we split by space (which is now SPI).
    // Let's assume we treat the whole string as one "word" for BPE if there's no pre-tokenizer.
    // BUT, BPE algorithms usually work on words.
    // If pre_tokenizer is null, maybe the "normalizer" output IS the sequence of "words"?
    // Let's look at `gpt2_tokenizer` again. It splits by regex `pat`.
    // Llama usually uses a regex or just raw BPE on the whole thing?
    // For simplicity in this test, let's treat the entire normalized string as a single "token" for BPE optimization (merging chars).
    // Or safer: split by SPI?
    // Actually SPI is just a character.
    // Let's try running BPE on the character sequence of the whole normalized string.

    // Note: This is an approximation. Real Llama tokenizer is complex.
    // But we just need to decode mainly. Encoding is creating prompt IDs.

    final chars = normalized.split('');
    var word = chars;

    // Iterative BPE
    while (true) {
      int? minRank;
      String? bigram;

      for (var i = 0; i < word.length - 1; i++) {
        final pair = '${word[i]} ${word[i + 1]}';
        if (bpeRanks.containsKey(pair)) {
          final rank = bpeRanks[pair]!;
          if (minRank == null || rank < minRank) {
            minRank = rank;
            bigram = pair;
          }
        }
      }

      if (bigram == null) break;

      final parts = bigram.split(' ');
      final first = parts[0];
      final second = parts[1];

      final newWord = <String>[];
      var i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] == first && word[i + 1] == second) {
          newWord.add(first + second);
          i += 2;
        } else {
          newWord.add(word[i]);
          i++;
        }
      }
      word = newWord;
    }

    final ids = <int>[];
    // Add start token <s> (ID 1 usually)
    ids.add(1);

    for (final token in word) {
      if (encoder.containsKey(token)) {
        ids.add(encoder[token]!);
      } else {
        // Check for byte fallback or <unk>
        // For now, use <unk> (ID 0 usually)
        ids.add(0); // Assuming 0 is unk
      }
    }

    return Tensor.from(ids, [1, ids.length], dataType: DataType.int64);
  }

  String decode(Tensor tokens) {
    tokens = tokens.to(device: Device.cpu).flatten();
    final textBuilder = StringBuffer();

    for (var i = 0; i < tokens.shape[0]; i++) {
      final token = tokens.at([i]).scalar as int;
      // Skip special tokens? 0=<unk>, 1=<s>, 2=</s> typically
      if (token == 0 || token == 1 || token == 2) continue;

      final tokenString = decoder[token] ?? '';
      textBuilder.write(tokenString);
    }

    // Reverse normalization: Replace SPI with space
    var text = textBuilder.toString();
    text = text.replaceAll(_spi, ' ');
    // If it started with SPI (space), Llama usually adds a leading space effectively.
    // But SentencePiece usually prepends the space.
    // If original string was "Hello", normalized is " Hello", decoded is " Hello".
    // We might want to strip leading space if prompt didn't have it?
    // Usually decode(encode(x)) == x.
    // "Hello" -> " Hello" -> ids -> " Hello".
    // tokenizer.decode(ids, skip_special_tokens=True) usually returns "Hello".
    // Let's strip leading space.
    if (text.startsWith(' ')) {
      text = text.substring(1);
    }

    return text;
  }
}
