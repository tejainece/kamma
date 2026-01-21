import 'package:kamma/kamma.dart';
import 'package:path/path.dart' as path;
import 'package:universal_io/universal_io.dart';

export 'gpt2_config.dart';
export 'gpt2_attention.dart';
export 'gpt2_mlp.dart';
export 'gpt2_block.dart';
export 'gpt2_model.dart';
export 'gpt2_lm_head_model.dart';
export 'gpt2_tokenizer.dart';
export 'attention_methods.dart';

class GPT2 extends Module {
  final GPT2Tokenizer tokenizer;
  final GPT2LMHeadModel model;

  GPT2({super.name = '', required this.tokenizer, required this.model});

  @override
  final Map<String, dynamic> meta = const {};

  @override
  final Iterable<Tensor> parameters = const [];

  @override
  void resetParameters() {
    model.resetParameters();
  }

  @override
  late final Iterable<Module> submodules = [model];

  // TODO sync with the transformers library implementation
  String prompt(
    String prompt, {
    required Context context,
    int maxNewTokens = 20,
    double temperature = 1.0,
    int topK = 0,
    double topP = 1.0,
  }) {
    final inputIds = tokenizer.encode(prompt);
    final outputIds = model.generate(
      inputIds,
      maxNewTokens: maxNewTokens,
      temperature: temperature,
      topK: topK,
      topP: topP,
      context: context,
    );
    return tokenizer.decode(outputIds);
  }

  static Future<GPT2> loadFromDirectory(
    String modelDir, {
    String prefix = '',
  }) async {
    final configFile = File(path.join(modelDir, 'config.json'));
    final config = await GPT2Config.fromFile(configFile);
    final tokenizer = await GPT2Tokenizer.fromPretrained(modelDir);
    final safeTensorFile = await SafeTensorsFile.load(
      path.join(modelDir, 'model.safetensors'),
    );
    final loader = safeTensorFile.mmapTensorLoader();
    final model = await GPT2LMHeadModel.loadFromSafeTensor(
      loader,
      config: config,
    );
    return GPT2(tokenizer: tokenizer, model: model);
  }
}
