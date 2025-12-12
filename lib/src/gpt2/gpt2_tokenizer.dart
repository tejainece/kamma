import 'dart:convert';
import 'dart:io';
import 'package:tensor/tensor.dart';
import 'package:path/path.dart' as path;

/// BPE tokenizer for GPT-2
class GPT2Tokenizer {
  final Map<String, int> encoder;
  final Map<int, String> decoder;
  final Map<String, int> bpeRanks;
  final Map<String, String> cache = {};
  final RegExp pat = RegExp(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    unicode: true,
  );

  GPT2Tokenizer({
    required this.encoder,
    required this.decoder,
    required this.bpeRanks,
  });

  static Map<int, String> bytesToUnicode() {
    final bs = <int>[
      ...List.generate(126 - 33 + 1, (i) => '!'.codeUnitAt(0) + i),
      ...List.generate(172 - 161 + 1, (i) => '¡'.codeUnitAt(0) + i),
      ...List.generate(255 - 174 + 1, (i) => '®'.codeUnitAt(0) + i),
    ];
    final cs = List<int>.from(bs);
    var n = 0;
    for (var b = 0; b < 256; b++) {
      if (!bs.contains(b)) {
        bs.add(b);
        cs.add(256 + n);
        n++;
      }
    }
    final csChars = cs.map((c) => String.fromCharCode(c)).toList();
    return Map.fromIterables(bs, csChars);
  }

  static Future<GPT2Tokenizer> fromPretrained(String modelPath) async {
    final vocabFile = File(path.join(modelPath, 'vocab.json'));
    final mergesFile = File(path.join(modelPath, 'merges.txt'));

    if (!await vocabFile.exists()) {
      throw Exception('vocab.json not found at $modelPath');
    }
    if (!await mergesFile.exists()) {
      throw Exception('merges.txt not found at $modelPath');
    }

    final vocabContent = await vocabFile.readAsString();
    final encoder = Map<String, int>.from(json.decode(vocabContent));
    final decoder = encoder.map((k, v) => MapEntry(v, k));

    final mergesContent = await mergesFile.readAsString();
    final mergesLines = mergesContent.split('\n');
    // Skip version comment if present (usually first line)
    final startIdx = mergesLines.first.startsWith('#') ? 1 : 0;

    final bpeRanks = <String, int>{};
    for (var i = startIdx; i < mergesLines.length; i++) {
      final line = mergesLines[i].trim();
      if (line.isEmpty) continue;
      // Merges are usually "token1 token2"
      // We store them as "token1 token2" -> rank (index)
      bpeRanks[line] = i - startIdx;
    }

    return GPT2Tokenizer(
      encoder: encoder,
      decoder: decoder,
      bpeRanks: bpeRanks,
    );
  }

  Set<String> _getPairs(List<String> word) {
    final pairs = <String>{};
    var prevChar = word[0];
    for (var i = 1; i < word.length; i++) {
      pairs.add('$prevChar ${word[i]}');
      prevChar = word[i];
    }
    return pairs;
  }

  String _bpe(String token) {
    if (cache.containsKey(token)) {
      return cache[token]!;
    }

    var word = token.split('').toList();
    var pairs = _getPairs(word);

    if (pairs.isEmpty) {
      return token;
    }

    while (true) {
      // Find the pair with the lowest rank
      String? bigram;
      int? minRank;

      for (final pair in pairs) {
        if (bpeRanks.containsKey(pair)) {
          final rank = bpeRanks[pair]!;
          if (minRank == null || rank < minRank) {
            minRank = rank;
            bigram = pair;
          }
        }
      }

      if (bigram == null) {
        break;
      }

      final parts = bigram.split(' ');
      final first = parts[0];
      final second = parts[1];

      final newWord = <String>[];
      var i = 0;
      while (i < word.length) {
        final j = word.indexOf(first, i);
        if (j == -1) {
          newWord.addAll(word.sublist(i));
          break;
        }
        newWord.addAll(word.sublist(i, j));
        i = j;

        if (word[i] == first && i < word.length - 1 && word[i + 1] == second) {
          newWord.add(first + second);
          i += 2;
        } else {
          newWord.add(word[i]);
          i += 1;
        }
      }

      word = newWord;
      if (word.length == 1) {
        break;
      }
      pairs = _getPairs(word);
    }

    final result = word.join(' ');
    cache[token] = result;
    return result;
  }

  Tensor encode(String text) {
    final bpeTokens = <int>[];
    final byteEncoder = bytesToUnicode();

    for (final match in pat.allMatches(text)) {
      final token = match.group(0)!;
      final tokenBytes = utf8.encode(token);
      final tokenChars = tokenBytes.map((b) => byteEncoder[b]!).join('');

      final bpeToken = _bpe(tokenChars);
      final bpeTokenParts = bpeToken.split(' ');

      for (final part in bpeTokenParts) {
        if (encoder.containsKey(part)) {
          bpeTokens.add(encoder[part]!);
        } else {
          // Fallback or unknown handling?
          // GPT-2 usually has byte-level fallback so everything should be in vocab if vocab is complete
          // But if not found, we might skip or use <unk>
          // For now, let's assume it's in encoder or we skip
          print('Warning: Token "$part" not found in vocabulary');
        }
      }
    }

    return Tensor.from(bpeTokens, [
      1,
      bpeTokens.length,
    ], datatype: DataType.int64);
  }

  String decode(Tensor tokens) {
    tokens = tokens.to(device: Device.cpu).flatten();
    final textBuilder = StringBuffer();
    final byteDecoder = bytesToUnicode().map((k, v) => MapEntry(v, k));

    for (var i = 0; i < tokens.shape[0]; i++) {
      final token = tokens.at([i]).scalar as int;
      final tokenString = decoder[token] ?? '';
      textBuilder.write(tokenString);
    }

    final text = textBuilder.toString();
    final bytes = text.split('').map((c) => byteDecoder[c]!).toList();
    return utf8.decode(bytes, allowMalformed: true);
  }
}
