import 'package:kamma/kamma.dart';

void main() async {
  final context = Context.best();
  final modelDir = '/Users/tejag/projects/dart/ai/testdata/models/llm/gpt2';
  final model = await GPT2Prompter.loadFromDirectory(modelDir);
  print(model.prompt('Hello, my name is', context: context));
}
