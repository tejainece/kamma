import 'package:kamma/kamma.dart';
import 'package:test/test.dart';

class _EncodeTokenTest {
  final String token;
  final List<int> encoding;

  const _EncodeTokenTest(this.token, this.encoding);

  static const tests = [
    _EncodeTokenTest('dart', [19841]),
    _EncodeTokenTest('hello', [3306]),
    _EncodeTokenTest('unbelievably', [33781]),
  ];
}

class _EncodeTestWithPad {
  final String token;
  final List<int> encoding;
  final int? padSize;
  final String decoding;

  const _EncodeTestWithPad({
    required this.token,
    required this.encoding,
    required this.padSize,
    required this.decoding,
  });

  static const testsWithoutPad = [
    _EncodeTestWithPad(
      token: 'How is the weather today?',
      encoding: [49406, 829, 533, 518, 2237, 721, 286, 49407],
      padSize: null,
      decoding: '<|startoftext|>how is the weather today ? <|endoftext|>',
    ),
    _EncodeTestWithPad(
      token: 'How is the weather today, brother?',
      encoding: [49406, 829, 533, 518, 2237, 721, 267, 3157, 286, 49407],
      padSize: null,
      decoding:
          '<|startoftext|>how is the weather today , brother ? <|endoftext|>',
    ),
  ];

  static const testsWithPad = [
    _EncodeTestWithPad(
      token: 'How is the weather today, brother?',
      encoding: [
        49406,
        829,
        533,
        518,
        2237,
        721,
        267,
        3157,
        286,
        49407,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
      ],
      padSize: 77,
      decoding:
          '<|startoftext|>how is the weather today , brother ? <|endoftext|>!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',
    ),
  ];

  static const testsWithPadChop = [
    _EncodeTestWithPad(
      token: 'How is the weather today, brother?',
      encoding: [49406, 829, 533, 518, 2237, 721, 0],
      padSize: 7,
      decoding: '<|startoftext|>how is the weather today !',
    ),
  ];
}

void main() async {
  final clip2_1 = await CLIPTokenizer.loadFromFile(
    'data/bpe_simple_vocab_16e6.txt',
    config: ClipTextConfig.v2_1,
  );
  group('CLIPTransformer', () {
    test('encodeToken', () {
      for (final test in _EncodeTokenTest.tests) {
        final encoded = clip2_1.encodeToken(test.token);
        expect(encoded, test.encoding, reason: 'for token: ${test.token}');
      }
    });
    test('encodeWithPad', () {
      for (final test in _EncodeTestWithPad.testsWithoutPad) {
        final encoded = clip2_1.encodeWithPad(test.token);
        expect(encoded, test.encoding, reason: 'for token: ${test.token}');
      }
    });
    test('encodeWithPad.WithPad', () {
      for (final test in _EncodeTestWithPad.testsWithPad) {
        final encoded = clip2_1.encodeWithPad(
          test.token,
          padSize: test.padSize,
        );
        expect(encoded, test.encoding, reason: 'for token: ${test.token}');
      }
    });
    test('encodeWithPad.Chop', () {
      for (final test in _EncodeTestWithPad.testsWithPadChop) {
        final encoded = clip2_1.encodeWithPad(
          test.token,
          padSize: test.padSize,
        );
        expect(encoded, test.encoding, reason: 'for token: ${test.token}');
      }
    });
    test('decode', () {
      for (final test in _EncodeTestWithPad.testsWithoutPad) {
        final encoded = clip2_1.decode(test.encoding);
        expect(encoded, test.decoding, reason: 'for token: ${test.token}');
      }
    });
    test('decode.withPad', () {
      for (final test in _EncodeTestWithPad.testsWithPad) {
        final encoded = clip2_1.decode(test.encoding);
        expect(encoded, test.decoding, reason: 'for token: ${test.token}');
      }
    });
    test('decode.withPad.Chop', () {
      for (final test in _EncodeTestWithPad.testsWithPadChop) {
        final encoded = clip2_1.decode(test.encoding);
        expect(encoded, test.decoding, reason: 'for token: ${test.token}');
      }
    });
  });
}
