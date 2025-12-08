import 'package:kamma/kamma.dart';

void main(List<String> args) async {
  final context = Context.best();

  final modelDir = 'models/llm/gpt_oss';
  final modelFile = '$modelDir/model.safetensors';

  final file = await SafeTensorsFile.load(modelFile);
  final loader = file.mmapTensorLoader();

  print('Loading configuration...');
  final config = GptOssConfig(
    vocabSize: 50257,
    nPositions: 1024,
    embedDim: 768,
    nLayer: 12,
    nHead: 12,
  );

  print('Initializing GPT-5 model...');
  final model = await GptOssForCausalLM.loadFromSafeTensor(
    loader,
    config: config,
  );
  print('Weights loaded.');

  print('Loading tokenizer...');
  final tokenizer = await GPT2Tokenizer.fromPretrained(modelDir);

  final prompt = args.isNotEmpty ? args.join(' ') : "Hello, my name is";
  print('Prompt: "$prompt"');

  final inputIds = tokenizer.encode(prompt);
  model.to_(context.device);

  print('Generating text (temperature=0.8, topK=50)...');
  final outputIds = model.generate(
    inputIds,
    maxNewTokens: 30,
    context: context,
    temperature: 0.8,
    topK: 50,
  );

  final generatedText = tokenizer.decode(outputIds);
  print('\n--- Generated Text ---\n');
  print(generatedText);
  print('\n----------------------\n');
}
