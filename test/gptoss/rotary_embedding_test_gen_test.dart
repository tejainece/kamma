import 'dart:io';
import 'package:kamma/src/gpt_oss/rotrary_embedding.dart';
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

void main() {
  test('Verify GPT-OSS Rotary Embedding against Python reference', () async {
    final path =
        '/home/tejag/projects/dart/ai/tensor/test_gen/gpt_oss/gpt_oss_rotary_embedding.safetensors';
    final file = File(path);
    if (!await file.exists()) {
      fail(
        'Test data not found at $path. Please run the test generator first.',
      );
    }

    final loader = await SafeTensorsFile.load(path);
    final mmap = loader.mmapTensorLoader();

    // 1. Verify InvFreq match (Internal check)
    final invFreqRef = mmap.loadByName('inv_freq');
    final positionIds = mmap.loadByName('position_ids');
    final cosRef = mmap.loadByName('cos');
    final sinRef = mmap.loadByName('sin');
    final qRef = mmap.loadByName('q');
    final kRef = mmap.loadByName('k');
    final qOutRef = mmap.loadByName('q_out');
    final kOutRef = mmap.loadByName('k_out');

    // Expected Parameters from Python generator
    final dim = 64;
    final maxPositionEmbeddings = 2048;
    final base = 10000.0;
    final seqLen = 10;

    final rotaryEmb = GptOssRotaryEmbedding(
      name: 'rotary_emb',
      dim: dim,
      maxPositionEmbeddings: maxPositionEmbeddings,
      base: base,
    );

    // Check InvFreq
    expect(
      rotaryEmb.invFreq.allClose(invFreqRef, atol: 1e-5),
      isTrue,
      reason: 'Inverse frequencies should match',
    );

    // 2. Verify Forward (Cos/Sin generation)
    // Note: Python generator computed cos/sin for specific position_ids
    // Dart forward takes positionIds
    final result = rotaryEmb.forward(positionIds, seqLen: seqLen);

    expect(
      result.cos.allClose(cosRef, atol: 1e-5),
      isTrue,
      reason: 'Cos values should match',
    );
    expect(
      result.sin.allClose(sinRef, atol: 1e-5),
      isTrue,
      reason: 'Sin values should match',
    );

    // 3. Verify ApplyRotaryPosEmb
    // q and k are modified in place in Dart
    final qDart = qRef.clone();
    final kDart = kRef.clone();

    // In Dart we pass the cos/sin from forward (which we verified matches Ref)
    // Or we can pass Ref cos/sin to isolate testing applyRotaryPosEmb
    GptOssRotaryEmbedding.applyRotaryPosEmb(qDart, kDart, cosRef, sinRef);

    expect(
      qDart.allClose(qOutRef, atol: 1e-5),
      isTrue,
      reason: 'Rotated Q should match',
    );
    expect(
      kDart.allClose(kOutRef, atol: 1e-5),
      isTrue,
      reason: 'Rotated K should match',
    );

    mmap.release();
  });
}
