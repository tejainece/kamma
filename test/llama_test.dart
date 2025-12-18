import 'package:test/test.dart';
import 'package:tensor/tensor.dart';
import 'package:kamma/src/llama/llama_config.dart';
import 'package:kamma/src/llama/llama_model.dart';
import 'package:kamma/src/llama/llama_attention.dart';

void main() {
  test('LlamaConfig init', () {
    final config = LlamaConfig();
    expect(config.hiddenSize, 4096);
  });

  test('LlamaAttention forward pass', () {
    final config = LlamaConfig(
      hiddenSize: 32,
      numAttentionHeads: 4,
      numHiddenLayers: 2,
      intermediateSize: 64,
      vocabSize: 100,
      maxPositionEmbeddings: 128,
      ropeTheta: 10000.0,
    );

    final attention = LlamaAttention.make(config);
    final context = Context.best();

    // Create dummy input: (bsz, seq_len, hidden_size)
    final x = Tensor.randn([1, 10, 32]);
    final positionIds = Tensor.arange(0, 10).unsqueeze(0);
    // position embeddings (cos, sin) - need to generate them or mock?
    // LlamaAttention calculates RoPE inside if we pass positionEmbeddings (cos, sin).
    // Or we can manually pass them.
    // However, LlamaModel handles this. Let's test LlamaModel instead or manually gen.

    // For unit test of attention, we can mock or pass null if strict logic doesn't crash (it might crash if posEmbeddings null).
    // But LlamaAttention implementation: if (positionEmbeddings != null) ...
    // So it might run without RoPE if we don't pass them.

    final output = attention.forward(x, context: context);
    expect(output.shape, [1, 10, 32]);
  });

  test('LlamaModel forward pass', () {
    final config = LlamaConfig(
      hiddenSize: 32,
      numAttentionHeads: 4,
      numHiddenLayers: 2,
      intermediateSize: 64,
      vocabSize: 100,
      maxPositionEmbeddings: 128,
    );

    final model = LlamaForCausalLM.make(config);
    final context = Context.best();

    // Input IDs: (bsz, seq_len)
    final inputIds = Tensor.randint(100, [1, 10], datatype: DataType.int64);

    final output = model.forward(inputIds, context: context);

    // Output logits: (bsz, seq_len, vocab_size)
    expect(output.shape, [1, 10, 100]);
  });
}
