import 'package:tensor/tensor.dart';
import '../gpt2/gpt2_block.dart';
import 'decision_transformer_config.dart';

class DecisionTransformerOutput {
  final Tensor statePreds;
  final Tensor actionPreds;
  final Tensor returnPreds;
  final Tensor hiddenStates;

  DecisionTransformerOutput({
    required this.statePreds,
    required this.actionPreds,
    required this.returnPreds,
    required this.hiddenStates,
  });
}

class DecisionTransformerModel extends Module {
  final DecisionTransformerConfig config;
  final EmbeddingLayer embedTime;
  final LinearLayer embedState;
  final LinearLayer embedAction;
  final LinearLayer embedReturn;
  final Dropout drop;
  final List<GPT2Block> h;
  final LayerNorm lnF;
  final LinearLayer predictState;
  final LinearLayer predictAction;
  final LinearLayer predictReturn;

  DecisionTransformerModel({
    required super.name,
    required this.config,
    required this.embedTime,
    required this.embedState,
    required this.embedAction,
    required this.embedReturn,
    required this.drop,
    required this.h,
    required this.lnF,
    required this.predictState,
    required this.predictAction,
    required this.predictReturn,
  });

  DecisionTransformerOutput forward(
    Tensor states,
    Tensor actions,
    Tensor returns,
    Tensor timesteps, {
    Tensor? attentionMask,
    required Context context,
  }) {
    context.onloadModule(this);

    final batchSize = states.shape[0];
    final seqLen = states.shape[1];

    if (attentionMask == null) {
      attentionMask = Tensor.ones(
        [batchSize, seqLen],
        datatype: DataType.float32,
        device: states.device,
      );
    }

    // Embeddings
    final timeEmbeddings = embedTime.forward(timesteps, context: context);
    final stateEmbeddings =
        embedState.forward(states, context: context) + timeEmbeddings;
    final actionEmbeddings =
        embedAction.forward(actions, context: context) + timeEmbeddings;
    final returnEmbeddings =
        embedReturn.forward(returns, context: context) + timeEmbeddings;

    // Stack embeddings: (batch, seq_len, 3, hidden_size) -> (batch, 3 * seq_len, hidden_size)
    // We want the order: R_t, s_t, a_t
    // So we stack [returns, states, actions] along dim 1
    // But we need to interleave them.
    // (batch, seq_len, hidden_size)

    // Efficient interleaving:
    // Stack along a new dimension 1: (batch, 3, seq_len, hidden_size)
    var stackedInputs = Tensor.cat([
      returnEmbeddings.unsqueeze(1),
      stateEmbeddings.unsqueeze(1),
      actionEmbeddings.unsqueeze(1),
    ], dim: 1);
    // Permute to (batch, seq_len, 3, hidden_size)
    stackedInputs = stackedInputs.permute([0, 2, 1, 3]);
    // Reshape to (batch, 3 * seq_len, hidden_size)
    stackedInputs = stackedInputs.reshape([
      batchSize,
      3 * seqLen,
      config.nEmbd,
    ]);

    var hiddenStates = drop.forward(stackedInputs, context: context);

    // Adjust attention mask for the stacked sequence
    // The original mask is (batch, seq_len)
    // We need (batch, 3 * seq_len)
    // We repeat each mask value 3 times
    var stackedAttentionMask = Tensor.cat([
      attentionMask.unsqueeze(1),
      attentionMask.unsqueeze(1),
      attentionMask.unsqueeze(1),
    ], dim: 1);
    stackedAttentionMask = stackedAttentionMask.permute([0, 2, 1]);
    stackedAttentionMask = stackedAttentionMask.reshape([
      batchSize,
      3 * seqLen,
    ]);

    // Apply GPT-2 blocks
    for (final block in h) {
      hiddenStates = block.forward(
        hiddenStates,
        attentionMask: stackedAttentionMask,
        context: context,
      );
    }

    hiddenStates = lnF.forward(hiddenStates, context: context);

    // We need to unstack the hidden states to get predictions
    // (batch, 3 * seq_len, hidden_size) -> (batch, seq_len, 3, hidden_size)
    hiddenStates = hiddenStates.reshape([batchSize, seqLen, 3, config.nEmbd]);
    // Permute back to (batch, 3, seq_len, hidden_size) if needed, or just slice
    // R_t is at index 0, s_t at 1, a_t at 2

    // Predict return from s_t (index 1) ? No, usually:
    // R_t is fed in.
    // s_t is fed in.
    // a_t is fed in.

    // Standard Decision Transformer:
    // Input: R_1, s_1, a_1, R_2, s_2, a_2 ...
    // Output at R_1 -> predict s_1? No.
    // Output at s_1 -> predict a_1.
    // Output at a_1 -> predict R_2?

    // Actually, let's check the paper/implementation details.
    // "we pass the sequence of tokens (R_1, s_1, a_1, ..., R_T, s_T, a_T) into a GPT architecture"
    // "To predict the action a_t, we use the token embedding corresponding to s_t"

    // So:
    // hiddenStates[:, :, 1, :] corresponds to s_t processing. We use this to predict a_t.
    // hiddenStates[:, :, 0, :] corresponds to R_t.
    // hiddenStates[:, :, 2, :] corresponds to a_t.

    // Let's verify standard implementation details.
    // HuggingFace:
    // x = x.reshape(batch_size, seq_length, 3, hidden_size).permute(0, 2, 1, 3)
    // return_preds = self.predict_return(x[:, 2])  // predict next return from state and action
    // state_preds = self.predict_state(x[:, 2])    // predict next state from state and action
    // action_preds = self.predict_action(x[:, 1])  // predict next action from state

    // My indices: 0=R, 1=s, 2=a.
    // So x[:, 1] is s_t. Correct.
    // x[:, 2] is a_t. Correct.

    // Re-extracting:
    // hiddenStates is (batch, seq_len, 3, hidden_size)
    final s_hidden = hiddenStates
        .index([Slice.all(), Slice.all(), 1])
        .squeeze(dim: 2); // Index 1
    final a_hidden = hiddenStates
        .index([Slice.all(), Slice.all(), 2])
        .squeeze(dim: 2); // Index 2
    // For return prediction, usually we predict next return? Or return at current step?
    // Standard implementation often predicts return from state (or action?).
    // Actually, let's look at HuggingFace again.
    // return_preds = self.predict_return(x[:, 2])  // predict next return from state and action (which is at index 2, a_t)
    // So it uses a_hidden.

    final actionPredsFinal = predictAction.forward(s_hidden, context: context);
    final statePredsFinal = predictState.forward(a_hidden, context: context);
    final returnPredsFinal = predictReturn.forward(a_hidden, context: context);

    return DecisionTransformerOutput(
      statePreds: statePredsFinal,
      actionPreds: actionPredsFinal,
      returnPreds: returnPredsFinal,
      hiddenStates: hiddenStates,
    );
  }

  @override
  void resetParameters() {
    embedTime.resetParameters();
    embedState.resetParameters();
    embedAction.resetParameters();
    embedReturn.resetParameters();
    // drop.resetParameters();
    for (final block in h) {
      block.resetParameters();
    }
    lnF.resetParameters();
    predictState.resetParameters();
    predictAction.resetParameters();
    predictReturn.resetParameters();
  }

  @override
  late final Iterable<Tensor> parameters = [
    ...embedTime.parameters,
    ...embedState.parameters,
    ...embedAction.parameters,
    ...embedReturn.parameters,
    ...h.expand((block) => block.parameters),
    ...lnF.parameters,
    ...predictState.parameters,
    ...predictAction.parameters,
    ...predictReturn.parameters,
  ];

  @override
  late final Iterable<Module> submodules = [
    embedTime,
    embedState,
    embedAction,
    embedReturn,
    drop,
    ...h,
    lnF,
    predictState,
    predictAction,
    predictReturn,
  ];

  @override
  Map<String, dynamic> get meta => {
    "config": config, // TODO: serialize config
  };

  static DecisionTransformerModel make({
    required DecisionTransformerConfig config,
    required String name,
  }) {
    final embedTime = EmbeddingLayer.make(
      config.maxEpLen,
      config.nEmbd,
      name: 'embed_time',
    );
    final embedState = LinearLayer.make(
      inFeatures: config.stateDim,
      outFeatures: config.nEmbd,
      name: 'embed_state',
    );
    final embedAction = LinearLayer.make(
      inFeatures: config.actDim,
      outFeatures: config.nEmbd,
      name: 'embed_action',
    );
    final embedReturn = LinearLayer.make(
      inFeatures: 1,
      outFeatures: config.nEmbd,
      name: 'embed_return',
    );

    final drop = Dropout(config.embdPdrop);

    final h = <GPT2Block>[];
    for (int i = 0; i < config.nLayer; i++) {
      h.add(GPT2Block.make(config: config, name: 'h.$i', layerIdx: i));
    }

    final lnF = LayerNorm.make(
      name: 'ln_f',
      normalizedShape: [config.nEmbd],
      eps: config.layerNormEpsilon,
    );

    final predictState = LinearLayer.make(
      inFeatures: config.nEmbd,
      outFeatures: config.stateDim,
      name: 'predict_state',
    );
    final predictAction = LinearLayer.make(
      inFeatures: config.nEmbd,
      outFeatures: config.actDim,
      name: 'predict_action',
      hasBias:
          false, // Usually tanh is applied after? Or just linear? HF uses tanh on action output sometimes.
    );
    final predictReturn = LinearLayer.make(
      inFeatures: config.nEmbd,
      outFeatures: 1,
      name: 'predict_return',
    );

    return DecisionTransformerModel(
      name: name,
      config: config,
      embedTime: embedTime,
      embedState: embedState,
      embedAction: embedAction,
      embedReturn: embedReturn,
      drop: drop,
      h: h,
      lnF: lnF,
      predictState: predictState,
      predictAction: predictAction,
      predictReturn: predictReturn,
    );
  }
}
