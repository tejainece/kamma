import 'dart:io';
import 'package:kamma/kamma.dart';

void main(List<String> args) async {
  // 1. Setup paths
  final modelDir = 'models/llm/gpt_oss';
  final modelFile = '$modelDir/model.safetensors';

  // 2. Load Configuration
  print('Loading configuration...');
  // Using standard GPT-2 small config
  final config = GptOssConfig(
    vocabSize: 50257,
    nPositions: 1024,
    nEmbd: 768,
    nLayer: 12,
    nHead: 12,
  );

  // 3. Initialize Model
  print('Initializing GPT-5 model...');
  final model = GptOssForCausalLM.make(config: config, name: 'gpt_oss');

  // 4. Load Weights
  print('Loading weights from $modelFile...');
  if (!File(modelFile).existsSync()) {
    print(
      'Error: Model file not found. Did you run "make -C models/llm/gpt_oss"?',
    );
    exit(1);
  }
  await model.loadFromSafeTensor(modelFile);
  print('Weights loaded.');

  // 5. Setup Tokenizer
  print('Loading tokenizer...');
  final tokenizer = await GPT2Tokenizer.fromPretrained(modelDir);

  // 6. Prepare Input
  final prompt = args.isNotEmpty ? args.join(' ') : "Hello, my name is";
  print('Prompt: "$prompt"');

  final inputIds = tokenizer.encode(prompt);
  final context = Context.best();
  model.to_(context.device); // Corrected from .to

  // 7. Generate
  print('Generating text (temperature=0.8, topK=50)...');
  final outputIds = model.generate(
    inputIds,
    maxNewTokens: 30,
    context: context,
    temperature: 0.8,
    topK: 50,
  );

  // 8. Decode Output
  final generatedText = tokenizer.decode(outputIds);
  print('\n--- Generated Text ---\n');
  print(generatedText);
  print('\n----------------------\n');
}
