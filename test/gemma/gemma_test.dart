import 'package:kamma/kamma.dart';
import 'package:test/test.dart';
import 'dart:io';
import 'dart:convert';

void main() {
  test('Gemma Inference Test', () async {
    final device = Device.cpu;
    final context = Context(device: device);

    final testDataDir = Directory(
      '/Users/tejag/projects/dart/ai/testdata/test_data/gemma',
    );
    if (!testDataDir.existsSync()) {
      print('Test data not found, skipping test.');
      return;
    }

    final configJson = jsonDecode(
      File('${testDataDir.path}/config.json').readAsStringSync(),
    );
    final config = GemmaConfig.fromJson(configJson);

    final stFile = await SafeTensorsFile.load(
      '${testDataDir.path}/model.safetensors',
    );
    final loader = stFile.cpuLoader();

    final model = await GemmaForCausalLM.loadFromSafeTensor(
      loader,
      config: config,
      activation: Activation.gelu, // Default for Gemma
      attentionMethod: GPT2AttentionMethodType.sdap, // Standard
    );

    final stFileInputs = await SafeTensorsFile.load(
      '${testDataDir.path}/inputs_outputs.safetensors',
    );
    final inputsOutputs = stFileInputs.cpuLoader();
    final inputIds = await inputsOutputs.loadByName('input_ids');
    final expectedLogits = await inputsOutputs.loadByName('logits');

    final output = model.forward(inputIds, context: context);

    final logits = output.logits;

    expect(logits.allClose(expectedLogits, atol: 1e-3, rtol: 1e-3), isTrue);
  });
}
