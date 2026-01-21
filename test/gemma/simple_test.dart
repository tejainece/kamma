import 'dart:convert';
import 'dart:io';
import 'package:kamma/kamma.dart';
import 'package:kamma/src/gemma/gemma_config.dart';
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';
import 'package:path/path.dart' as path;

void main() async {
  print('Starting simple test...');
  Context context = Context.best();
  print('Using device: ${context.device}');

  final modelDir = './../testdata/models/llm/gemma/v1/2b';
  final configFile = File(path.join(modelDir, 'config.json'));
  final configJson = json.decode(await configFile.readAsString());
  var config = GemmaConfig.fromJson(configJson);

  print('Loading model...');
  final loader = await CompositeSafeTensorLoader.loadSplitSafeTensors(
    Directory(modelDir),
  );

  final model = await GemmaForCausalLM.loadFromSafeTensor(
    loader,
    config: config,
    activation: Activation.gelu,
    attentionMethod: GPT2AttentionMethodType.sdap,
  );
  print('Model loaded.');

  final input = Tensor.zeros(
    [1, 10],
    dataType: DataType.int64,
    device: context.device,
  );
  print('Running forward pass...');
  final result = model.forward(input, context: context);
  print('Forward pass done. Logits shape: ${result.logits.shape}');
}
