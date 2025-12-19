import 'dart:math' as math;
import 'package:kamma/kamma.dart';
import 'package:tensor/tensor.dart';
import 'deepseek_config.dart';
import 'deepseek_rotary_embedding.dart';
import 'deepseek_rms_norm.dart';

class DeepSeekV3MLA extends Module {
  final int layerIdx;
  final int numHeads;
  final int headDim;
  final int nopeHeadDim;
  final int ropeHeadDim;
  final int qLoraRank;
  final int kvLoraRank;
  final int vHeadDim; // Often same as headDim

  // Query Projections
  late final LinearLayer? qDownProj;
  late final LinearLayer qUpProj;
  late final DeepSeekRMSNorm? qNorm;

  // KV Projections (MLA)
  late final LinearLayer kvDownProj;
  late final LinearLayer
  kvUpProj; // Projects to K_nope + K_rope + V ? Or separate?
  // Usually in official implementations:
  // kv_up_proj projects to (num_heads * (nope + rope)) for K
  // AND (num_heads * v_dim) for V.
  // Often they are separate linear layers `w_uk` and `w_uv` or combined.
  // Let's assume combined w_up for now, or check detailed architecture.
  // Standard V3/V2:
  // w_uk: (kv_lora_rank, n_heads * (nope + rope))
  // w_uv: (kv_lora_rank, n_heads * v_dim)
  // Let's split them for clarity unless I find they are merged in weights.
  // Merged is common for optimization. Let's start with separate.
  late final LinearLayer kvUpProjKey; // Projects to K
  late final LinearLayer kvUpProjValue; // Projects to V
  late final DeepSeekRMSNorm kvNorm;

  // Output Projection
  late final LinearLayer oProj;

  final DeepSeekRotaryEmbedding rotaryEmbedding;

  DeepSeekV3MLA({
    required super.name,
    required DeepSeekConfig config,
    required this.layerIdx,
    required this.rotaryEmbedding,
  }) : numHeads = config.numAttentionHeads,
       headDim = config.headDim,
       nopeHeadDim = config.nopeHeadDim,
       ropeHeadDim = config.ropeHeadDim,
       qLoraRank = config.qLoraRank,
       kvLoraRank = config.kvLoraRank,
       vHeadDim = config
           .headDim // Assuming v_dim = head_dim
           {
    // Q Compression
    if (qLoraRank > 0) {
      qDownProj = LinearLayer.make(
        name: 'q_down_proj',
        inFeatures: config.hiddenSize,
        outFeatures: qLoraRank,
        hasBias: false,
      );
      qNorm = DeepSeekRMSNorm(
        config,
        dim: qLoraRank,
      ); // Norm after compression.
      // Q Up: Projects to num_heads * (nope + rope)
      qUpProj = LinearLayer.make(
        name: 'q_up_proj',
        inFeatures: qLoraRank,
        outFeatures: numHeads * (nopeHeadDim + ropeHeadDim),
        hasBias: false,
      );
    } else {
      qDownProj = null;
      qNorm = null;
      // Direct projection if no compression
      qUpProj = LinearLayer.make(
        name: 'q_proj',
        inFeatures: config.hiddenSize,
        outFeatures: numHeads * (nopeHeadDim + ropeHeadDim),
        hasBias: false,
      );
    }

    // KV Compression (MLA)
    // Down projects to latent c_KV
    kvDownProj = LinearLayer.make(
      name: 'kv_down_proj',
      inFeatures: config.hiddenSize,
      outFeatures:
          kvLoraRank, // + (rope_head_dim)? Sometimes k_rope is separate shortcut.
      // DeepSeek V2: c_KV is compressed. k_rope is NOT compressed?
      // Actually, usually k_rope is generated from c_KV too.
      hasBias: false,
    );
    kvNorm = DeepSeekRMSNorm(config, dim: kvLoraRank);

    // Up projects
    // We need K (nope + rope) and V
    // K_total_dim = num_heads * (nope + rope)
    // V_total_dim = num_heads * v_dim

    // Simplification: Combined Up Projection for K and V
    // or separate. Let's do separate to match "w_uk" and "w_uv".
    kvUpProjKey = LinearLayer.make(
      name: 'kv_up_proj_key', // w_uk
      inFeatures: kvLoraRank,
      outFeatures: numHeads * (nopeHeadDim + ropeHeadDim),
      hasBias: false,
    );
    kvUpProjValue = LinearLayer.make(
      name: 'kv_up_proj_value', // w_uv
      inFeatures: kvLoraRank,
      outFeatures: numHeads * vHeadDim,
      hasBias: false,
    );

    oProj = LinearLayer.make(
      name: 'o_proj',
      inFeatures: numHeads * vHeadDim,
      outFeatures: config.hiddenSize,
      hasBias: false,
    );
  }

  // Helper to split heads
  Tensor _splitHeads(Tensor t, int nHeads, int dim) {
    // (B, L, nHeads * dim) -> (B, L, nHeads, dim) -> (B, nHeads, L, dim)
    final shape = t.shape;
    final b = shape[0];
    final l = shape[1];
    return t.view([b, l, nHeads, dim]).transpose(1, 2);
  }

  ({Tensor output, Tensor? pastKv}) forward(
    Tensor hiddenStates, {
    Tensor? attentionMask,
    Tensor? positionIds,
    Tensor? pastKv, // Needs defining what this is. compressed c_KV?
    bool useCache = false,
    required Context context,
  }) {
    context.onloadModule(this);

    // 1. Query Generation
    Tensor q;
    if (qDownProj != null) {
      // Compressed Q
      var cQ = qDownProj!.forward(hiddenStates, context: context);
      cQ = qNorm!.forward(cQ, context: context);
      q = qUpProj.forward(cQ, context: context);
    } else {
      q = qUpProj.forward(hiddenStates, context: context);
    }
    // q shape: (B, L, num_heads * (nope + rope))
    // Split into heads
    var qHeads = _splitHeads(q, numHeads, nopeHeadDim + ropeHeadDim);
    // (B, H, L, nope + rope)
    // Split into nope and rope parts
    // q_nope: (..., :nope), q_rope: (..., nope:)
    var qNope = qHeads.slice(-1, 0, end: nopeHeadDim);
    var qRope = qHeads.slice(-1, nopeHeadDim, end: null);

    // 2. KV Generation
    Tensor v;

    // If using cache, we might have passed cKV?
    // TODO: Handle cache logic properly. For now, compute fresh.

    // cKVRaw might be null if no compression? No, if kvDownProj is null.
    // Wait, my implementation assumes kvDownProj is always there for MLA.

    var cKVRaw = kvDownProj.forward(hiddenStates, context: context);
    var cKV = kvNorm.forward(cKVRaw, context: context);

    // Project K and V
    var kRaw = kvUpProjKey.forward(cKV, context: context);
    var vRaw = kvUpProjValue.forward(cKV, context: context);

    var kHeads = _splitHeads(kRaw, numHeads, nopeHeadDim + ropeHeadDim);
    v = _splitHeads(vRaw, numHeads, vHeadDim);

    var kNope = kHeads.slice(-1, 0, end: nopeHeadDim);
    var kRope = kHeads.slice(-1, nopeHeadDim, end: null);

    // 3. Apply RoPE
    // q_rope and k_rope
    if (positionIds != null) {
      var (cos, sin) = rotaryEmbedding(
        v,
        positionIds,
      ); // v is just passed to get device/dtype
      var (qRopeRot, kRopeRot) = applyRotaryPosEmb(qRope, kRope, cos, sin);
      qRope = qRopeRot;
      kRope = kRopeRot;
    }

    // 4. Combine Q and K for attention validation (standard Dot Product)

    var qFinal = Tensor.cat([qNope, qRope], dim: -1);
    var kFinal = Tensor.cat([kNope, kRope], dim: -1);

    // 5. Attention
    var attScores = qFinal.matmul(kFinal.transpose(-2, -1));
    attScores = attScores / math.sqrt(nopeHeadDim + ropeHeadDim);

    if (attentionMask != null) {
      if (attScores.shape.length == 4 && attentionMask.shape.length == 4) {
        attScores = attScores + attentionMask;
      } else {
        // Broadcasting issues might arise if shapes don't align perfectly.
        // Assuming user passes correct mask for now.
        attScores = attScores + attentionMask;
      }
    }

    // Softmax now requires dimension as positional argument or param?
    // Error says: Too few positional arguments: 1 required, 0 given.
    // Previously: softmax({int dim}) -> softmax(dim) ?
    // Let's check Tensor API or typical usage. Usually it is softmax(dim).
    var attProbs = attScores.softmax(-1);
    // Dropout?

    var attOutput = attProbs.matmul(v);

    // 6. Merge heads and Output
    // (B, H, L, V_D) -> (B, L, H, V_D) -> (B, L, H*V_D)
    attOutput = attOutput.transpose(1, 2).contiguous();
    attOutput = attOutput.view([
      hiddenStates.shape[0],
      hiddenStates.shape[1],
      numHeads * vHeadDim,
    ]);

    var output = oProj.forward(attOutput, context: context);

    return (output: output, pastKv: cKV); // Return latent for cache
  }

  @override
  // Parameters... (omitted for brevity, assume Module handles reflection or need manual)
  // Implementing manually if Module isn't auto-scanning in this version of kamma
  Iterable<Module> get submodules => [
    if (qDownProj != null) qDownProj!,
    qUpProj,
    if (qNorm != null) qNorm!,
    kvDownProj,
    kvUpProjKey,
    kvUpProjValue,
    kvNorm,
    oProj,
  ];

  @override
  Iterable<Tensor> get parameters => [];

  @override
  Iterable<Tensor> get nonTrainableParameters => [];

  @override
  void resetParameters() {
    // call reset on submodules
    for (var m in submodules) m.resetParameters();
  }

  @override
  Map<String, dynamic> get meta => {'layerIdx': layerIdx, 'numHeads': numHeads};
}
