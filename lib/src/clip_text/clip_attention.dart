import 'package:kamma/kamma.dart';

/// Multi-headed attention from 'Attention Is All You Need' paper
class ClipAttention extends Module {
  final LinearLayer kProj;
  final LinearLayer vProj;
  final LinearLayer qProj;
  final LinearLayer outProj;
  final int numAttensionHeads;

  /// Size of each attention head
  final int headDim;
  final double scale;
  final double dropout;
  final AttentionFunction attentionFunction;

  ClipAttention({
    required super.name,
    required this.kProj,
    required this.vProj,
    required this.qProj,
    required this.outProj,
    required this.numAttensionHeads,
    required this.headDim,
    required this.scale,
    required this.dropout,
    required this.attentionFunction,
  });

  (Tensor, Tensor) forward(
    Tensor x, {
    Tensor? attentionMask,
    required Context context,
  }) {
    final [batchSize, seqLength, embedDim] = x.sizes;
    Tensor queries = qProj.forward(x, context: context);
    Tensor keys = kProj.forward(x, context: context);
    Tensor values = vProj.forward(x, context: context);

    queries = queries.view([batchSize, seqLength, -1, headDim]).transpose(1, 2);
    keys = keys.view([batchSize, seqLength, -1, headDim]).transpose(1, 2);
    values = values.view([batchSize, seqLength, -1, headDim]).transpose(1, 2);

    var (attentionOutput, attentionWeights) = attentionFunction.forward(
      queries,
      keys,
      values,
      attentionMask: attentionMask,
      scaling: scale,
      dropout: context.isTraining ? dropout : 0,
    );

    attentionOutput = attentionOutput.reshape([
      batchSize,
      seqLength,
      -1,
    ]).contiguous();
    attentionOutput = outProj.forward(attentionOutput, context: context);

    return (attentionOutput, attentionWeights);
  }

  int get embedDim => numAttensionHeads * headDim;

  @override
  void resetParameters() {
    kProj.resetParameters();
    vProj.resetParameters();
    qProj.resetParameters();
    outProj.resetParameters();
  }

  @override
  late final Map<String, dynamic> meta = {
    'kProj': kProj.meta,
    'vProj': vProj.meta,
    'qProj': qProj.meta,
    'outProj': outProj.meta,
    'numAttensionHeads': numAttensionHeads,
    'headDim': headDim,
    'scale': scale,
    'dropout': dropout,
    'attentionFunction': attentionFunction.runtimeType.toString(),
  };

  @override
  late final Iterable<Tensor> parameters = [];

  @override
  Iterable<Module> get submodules => [kProj, vProj, qProj, outProj];

  static Future<ClipAttention> loadFromSafeTensor(
    SafeTensorLoader loader, {
    String prefix = '',
    required String name,
    required ClipTextConfig config,
  }) async {
    final qProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}q_proj.',
    );
    final kProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}k_proj.',
    );
    final vProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}v_proj.',
    );
    final outProj = await LinearLayer.loadFromSafeTensor(
      loader,
      prefix: '${prefix}out_proj.',
    );

    return ClipAttention(
      name: name,
      kProj: kProj,
      vProj: vProj,
      qProj: qProj,
      outProj: outProj,
      numAttensionHeads: 0, // TODO
      headDim: 0, // TODO
      scale: 0, // TODO
      dropout: config.attentionDropout,
      attentionFunction: EagerAttentionFunction(),
    );
  }
}

abstract class AttentionFunction {
  (Tensor, Tensor) forward(
    Tensor q,
    Tensor k,
    Tensor v, {
    Tensor? attentionMask,
    double dropout = 0,
    double? scaling,
  });
}

class EagerAttentionFunction implements AttentionFunction {
  @override
  (Tensor, Tensor) forward(
    Tensor q,
    Tensor k,
    Tensor v, {
    Tensor? attentionMask,
    double dropout = 0,
    double? scaling,
  }) {
    Tensor attentionWeights = q.matmul(k.transpose(-1, -2));
    if (scaling != null) {
      attentionWeights = attentionWeights * scaling;
    }
    if (attentionMask != null) {
      attentionWeights = attentionWeights + attentionMask;
    }

    attentionWeights = attentionWeights
        .softmax(-1, dataType: DataType.float32)
        .to(dataType: q.dataType);
    attentionWeights = NNUtil.dropout(
      attentionWeights,
      dropout,
      training: false, // TODO training: ,
    );

    Tensor output = attentionWeights.matmul(v);
    attentionWeights = attentionWeights.transpose(1, 2).contiguous();

    return (output, attentionWeights);
  }
}
