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

class GPT2Prompter extends Module {
  final GPT2Tokenizer tokenizer;
  final GPT2LMHeadModel model;

  GPT2Prompter({super.name = '', required this.tokenizer, required this.model});

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
  String prompt(String prompt, {required Context context}) {
    // TODO perform this inside model?
    model.resetKeyValueCache();

    final inputIds = tokenizer.encode(prompt);
    final outputIds = model.generate(
      inputIds,
      maxNewTokens: 20, // TODO
      temperature: 0.0, // TODO
      context: context,
    );
    return tokenizer.decode(outputIds);
  }

  static Future<GPT2Prompter> loadFromDirectory(
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
    return GPT2Prompter(tokenizer: tokenizer, model: model);
  }
}
