import 'package:kamma/kamma.dart';
import 'gpt1_attention.dart';
import 'gpt1_mlp.dart';

class OpenAIGPTBlock extends Module {
  final OpenAIGPTAttention attn;
  final LayerNorm ln1;
  final OpenAIGPTMLP mlp;
  final LayerNorm ln2;

  OpenAIGPTBlock({
    required super.name,
    required this.attn,
    required this.ln1,
    required this.mlp,
    required this.ln2,
  });

  Tensor forward(
    Tensor x, {
    Tensor? attentionMask,
    Tensor? headMask,
    bool outputAttentions = false,
    required Context context,
  }) {
    context.onloadModule(this);

    // Attention
    Tensor attnOutput = attn.forward(
      x,
      attentionMask: attentionMask,
      headMask: headMask,
      outputAttentions: outputAttentions,
      context: context,
    );

    // Add + Norm 1 (Post-Norm)
    Tensor n = ln1.forward(x + attnOutput, context: context);

    // MLP
    Tensor mlpOutput = mlp.forward(n, context: context);

    // Add + Norm 2 (Post-Norm)
    Tensor h = ln2.forward(n + mlpOutput, context: context);

    return h;
  }

  @override
  void resetParameters() {
    attn.resetParameters();
    ln1.resetParameters();
    mlp.resetParameters();
    ln2.resetParameters();
  }

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  late final Iterable<Module> submodules = [attn, ln1, mlp, ln2];

  @override
  Map<String, dynamic> get meta => {};

  static OpenAIGPTBlock make({
    required String name,
    required int nCtx,
    required int nEmbd,
    required int nHead,
    required double attnPdrop,
    required double residPdrop,
    required double layerNormEpsilon,
    String afn = "gelu",
  }) {
    final attn = OpenAIGPTAttention.make(
      name: 'attn',
      nCtx: nCtx,
      nEmbd: nEmbd,
      nHead: nHead,
      attnPdrop: attnPdrop,
      residPdrop: residPdrop,
      scale: true,
    );

    final ln1 = LayerNorm.make(
      name: 'ln_1',
      normalizedShape: [nEmbd],
      eps: layerNormEpsilon,
    );

    final mlp = OpenAIGPTMLP.make(
      name: 'mlp',
      nEmbd: nEmbd,
      residPdrop: residPdrop,
      afn: afn,
    );

    final ln2 = LayerNorm.make(
      name: 'ln_2',
      normalizedShape: [nEmbd],
      eps: layerNormEpsilon,
    );

    return OpenAIGPTBlock(name: name, attn: attn, ln1: ln1, mlp: mlp, ln2: ln2);
  }

  static Future<OpenAIGPTBlock> loadFromSafeTensor(
    SafeTensorLoader loader, {
    required String prefix,
    required String name,
    required int nCtx,
    required int nEmbd,
    required int nHead,
    required double attnPdrop,
    required double residPdrop,
    required double layerNormEpsilon,
    String afn = "gelu",
  }) async {
    final attn = await OpenAIGPTAttention.loadFromSafeTensor(
      loader,
      prefix: '${prefix}attn.',
      name: 'attn',
      nCtx: nCtx,
      nEmbd: nEmbd,
      nHead: nHead,
      attnPdrop: attnPdrop,
      residPdrop: residPdrop,
      scale: true,
    );

    final ln1 = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}ln_1.',
      name: 'ln_1',
      normalizedShape: [nEmbd],
    );

    final mlp = await OpenAIGPTMLP.loadFromSafeTensor(
      loader,
      prefix: '${prefix}mlp.',
      name: 'mlp',
      residPdrop: residPdrop,
      afn: afn,
    );

    final ln2 = await LayerNorm.loadFromSafeTensor(
      loader,
      prefix: '${prefix}ln_2.',
      name: 'ln_2',
      normalizedShape: [nEmbd],
    );

    return OpenAIGPTBlock(name: name, attn: attn, ln1: ln1, mlp: mlp, ln2: ln2);
  }
}
