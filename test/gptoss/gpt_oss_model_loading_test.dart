import 'dart:convert';
import 'dart:io';

import 'package:kamma/kamma.dart';
import 'package:kamma/src/gpt2/gpt2_tokenizer.dart';
import 'package:path/path.dart' as path;
import 'package:tensor/tensor.dart';
import 'package:test/test.dart';

// Helper to load tokenizer from huggingface tokenizer.json
Future<GPT2Tokenizer> loadTokenizerFromJson(String path) async {
  final file = File(path);
  if (!await file.exists()) {
    throw Exception('tokenizer.json not found at $path');
  }
  final jsonContent = await file.readAsString();
  final jsonMap = json.decode(jsonContent) as Map<String, dynamic>;

  // Structure: { "model": { "vocab": { "token": id }, "merges": ["u g", ...] } }
  final model = jsonMap['model'] as Map<String, dynamic>;
  final vocab = model['vocab'] as Map<String, dynamic>;
  final mergesList = model['merges'] as List;
  final merges = mergesList.map((e) {
    if (e is String) return e;
    if (e is List) return e.join(' ');
    return e.toString();
  }).toList();

  final encoder = vocab.map((k, v) => MapEntry(k, v as int));
  final decoder = encoder.map((k, v) => MapEntry(v, k));
  final bpeRanks = <String, int>{};
  for (int i = 0; i < merges.length; i++) {
    bpeRanks[merges[i]] = i;
  }

  return GPT2Tokenizer(encoder: encoder, decoder: decoder, bpeRanks: bpeRanks);
}

// Helper to handle sharded safetensors based on index.json
class ShardedSafeTensorLoader extends SafeTensorLoader {
  final Directory baseDir;
  final Map<String, String> indexMap;
  final Map<String, SafeTensorLoader> _loaders = {};

  ShardedSafeTensorLoader({required this.baseDir, required this.indexMap});

  @override
  SafeTensorHeader get header =>
      throw UnimplementedError('Sharded loader does not have a single header');

  Future<SafeTensorLoader> _getLoader(String name) async {
    final filename = indexMap[name];
    if (filename == null) {
      throw Exception('Tensor $name not found in index');
    }

    if (_loaders.containsKey(filename)) {
      return _loaders[filename]!;
    }

    final filePath = path.join(baseDir.path, filename);
    final loader = await SafeTensorsFile.load(filePath);
    final mmapLoader = loader.mmapTensorLoader();
    _loaders[filename] = mmapLoader;
    return mmapLoader;
  }

  @override
  bool hasTensor(String name) => indexMap.containsKey(name);

  @override
  bool hasTensorWithPrefix(String prefix) {
    for (final key in indexMap.keys) {
      if (key.startsWith(prefix)) return true;
    }
    return false;
  }

  @override
  Future<Tensor> loadByName(String name, {Device device = Device.cpu}) async {
    final loader = await _getLoader(name);
    return loader.loadByName(name, device: device);
  }

  @override
  Future<Tensor?> tryLoadByName(
    String name, {
    Device device = Device.cpu,
  }) async {
    if (!hasTensor(name)) return null;
    final loader = await _getLoader(name);
    return loader.tryLoadByName(name, device: device);
  }

  void release() {
    for (final loader in _loaders.values) {
      if (loader is MmapSafeTensorLoader) {
        loader.release();
      }
    }
  }
}

void main() {
  final modelDir = '/home/tejag/projects/dart/ai/tensor/models/llm/gpt_oss';

  test('Load GPT-OSS model and run generation', () async {
    print('Loading config...');
    final configFile = File(path.join(modelDir, 'config.json'));
    final configJson = json.decode(await configFile.readAsString());
    final config = GptOssConfig.fromJson(configJson);

    print('Loading tokenizer...');
    final tokenizer = await loadTokenizerFromJson(
      path.join(modelDir, 'tokenizer.json'),
    );

    print('Loading model index...');
    final indexFile = File(path.join(modelDir, 'model.safetensors.index.json'));
    final indexJson = json.decode(await indexFile.readAsString());
    final weightMap = (indexJson['weight_map'] as Map<String, dynamic>)
        .cast<String, String>();

    print('Initializing loader...');
    final loader = ShardedSafeTensorLoader(
      baseDir: Directory(modelDir),
      indexMap: weightMap,
    );

    print('Loading model weights (this may take time)...');
    final context = Context.best();
    print('Using device: ${context.device}');

    final model = await GptOssForCausalLM.loadFromSafeTensor(
      loader,
      config: config,
      name: 'gpt_oss',
    );

    // Move model to device
    model.to_(context.device);

    print('Model loaded. Running inference...');

    final prompt = "The future of AI is";
    print('Prompt: "$prompt"');

    final inputIds = tokenizer.encode(prompt).to(device: context.device);

    final generatedIds = model.generate(
      inputIds,
      maxNewTokens: 20,
      context: context,
      temperature: 0.7, // Optional if implemented
      topK: 50, // Optional if implemented
    );

    final outputText = tokenizer.decode(generatedIds);
    print('Output: "$outputText"');

    expect(outputText, startsWith(prompt));
    expect(generatedIds.shape[1], greaterThan(inputIds.shape[1]));

    // Cleanup
    loader.release();
  }, timeout: Timeout(Duration(minutes: 5)));
}
