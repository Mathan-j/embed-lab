/// Thin HTTP client over the embed-lab backend's five endpoints, plus /api/demos.
///
/// Base URL resolution:
///   1. `--dart-define=API_BASE=...` always wins if given.
///   2. Otherwise, on Android this defaults to `http://10.0.2.2:8100` -- the
///      Android emulator's special alias for the host machine's `localhost`.
///      `localhost` inside the emulator means the emulator itself, not the
///      machine running it, so a plain `localhost` default would silently
///      fail on Android and work everywhere else.
///   3. Every other platform (web, Windows, desktop Chrome/Edge) defaults to
///      `http://localhost:8100`.
library;

import 'dart:convert';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:http/http.dart' as http;

// dart:io is not available on web; guard every use behind `!kIsWeb`.
// ignore: avoid_web_libraries_in_flutter
import 'dart:io' show Platform;

const String _apiBaseOverride = String.fromEnvironment('API_BASE');

String resolveBaseUrl() {
  if (_apiBaseOverride.isNotEmpty) return _apiBaseOverride;
  if (!kIsWeb) {
    try {
      if (Platform.isAndroid) return 'http://10.0.2.2:8100';
    } catch (_) {
      // Platform is unavailable in some embedder contexts; fall through.
    }
  }
  return 'http://localhost:8100';
}

/// A 409 from the backend: a user-fixable input error with a readable hint.
/// The whole point of the backend returning a hint instead of a stack trace
/// is that the UI shows it verbatim rather than a generic "something broke".
class ApiHintException implements Exception {
  final String detail;
  final String hint;
  ApiHintException(this.detail, this.hint);

  @override
  String toString() => '$detail ($hint)';
}

class TokenizeResult {
  final List<String> pieces;
  final List<int> ids;
  final int count;
  TokenizeResult({required this.pieces, required this.ids, required this.count});
  factory TokenizeResult.fromJson(Map<String, dynamic> j) => TokenizeResult(
        pieces: List<String>.from(j['pieces']),
        ids: List<int>.from(j['ids']),
        count: j['count'],
      );
}

class EmbedResult {
  final List<double> vector;
  final int dims;
  final double norm;
  EmbedResult({required this.vector, required this.dims, required this.norm});
  factory EmbedResult.fromJson(Map<String, dynamic> j) => EmbedResult(
        vector: List<double>.from(j['vector'].map((e) => (e as num).toDouble())),
        dims: j['dims'],
        norm: (j['norm'] as num).toDouble(),
      );
}

class NeighbourHit {
  final String word;
  final String category;
  final double score;
  NeighbourHit({required this.word, required this.category, required this.score});
  factory NeighbourHit.fromJson(Map<String, dynamic> j) => NeighbourHit(
        word: j['word'],
        category: j['category'],
        score: (j['score'] as num).toDouble(),
      );
}

class NeighboursResult {
  final List<NeighbourHit> hits;
  NeighboursResult(this.hits);
  factory NeighboursResult.fromJson(Map<String, dynamic> j) => NeighboursResult(
        List<Map<String, dynamic>>.from(j['hits']).map(NeighbourHit.fromJson).toList(),
      );
}

class CategoryProbability {
  final String category;
  final double probability;
  CategoryProbability({required this.category, required this.probability});
  factory CategoryProbability.fromJson(Map<String, dynamic> j) => CategoryProbability(
        category: j['category'],
        probability: (j['probability'] as num).toDouble(),
      );
}

class ClassifyResult {
  final String predicted;
  final List<CategoryProbability> all;
  final double macroF1;
  final double baselineMacroF1;
  final int seed;
  ClassifyResult({
    required this.predicted,
    required this.all,
    required this.macroF1,
    required this.baselineMacroF1,
    required this.seed,
  });
  factory ClassifyResult.fromJson(Map<String, dynamic> j) => ClassifyResult(
        predicted: j['predicted'],
        all: List<Map<String, dynamic>>.from(j['all']).map(CategoryProbability.fromJson).toList(),
        macroF1: (j['macro_f1'] as num).toDouble(),
        baselineMacroF1: (j['baseline_macro_f1'] as num).toDouble(),
        seed: j['seed'],
      );
}

class MapResult {
  final List<String> words;
  final List<String> categories;
  final List<int> clusters;
  final List<List<double>> coords;
  final int nClusters;
  final double ari;
  final int seed;
  MapResult({
    required this.words,
    required this.categories,
    required this.clusters,
    required this.coords,
    required this.nClusters,
    required this.ari,
    required this.seed,
  });
  factory MapResult.fromJson(Map<String, dynamic> j) => MapResult(
        words: List<String>.from(j['words']),
        categories: List<String>.from(j['categories']),
        clusters: List<int>.from(j['clusters']),
        coords: List<List<double>>.from(
          j['coords'].map((p) => List<double>.from(p.map((v) => (v as num).toDouble()))),
        ),
        nClusters: j['n_clusters'],
        ari: (j['ari'] as num).toDouble(),
        seed: j['seed'],
      );
}

class SubwordLeakagePair {
  final String plain;
  final String compound;
  SubwordLeakagePair({required this.plain, required this.compound});
  factory SubwordLeakagePair.fromJson(Map<String, dynamic> j) =>
      SubwordLeakagePair(plain: j['plain'], compound: j['compound']);
}

class DemosResult {
  final List<String> ambiguous;
  final List<SubwordLeakagePair> subwordLeakage;
  DemosResult({required this.ambiguous, required this.subwordLeakage});
  factory DemosResult.fromJson(Map<String, dynamic> j) => DemosResult(
        ambiguous: List<String>.from(j['ambiguous']),
        subwordLeakage: List<Map<String, dynamic>>.from(j['subword_leakage'])
            .map(SubwordLeakagePair.fromJson)
            .toList(),
      );
}

class EmbedLabApi {
  final String baseUrl;

  /// Optional client, for tests. `flutter test` installs an HttpOverrides that
  /// answers every request with a canned 400 and never touches the network, so
  /// a test cannot produce a real ClientException by pointing at a dead port --
  /// it has to inject a client that raises one.
  final http.Client _client;

  EmbedLabApi({String? baseUrl, http.Client? client})
      : baseUrl = baseUrl ?? resolveBaseUrl(),
        _client = client ?? http.Client();

  Future<Map<String, dynamic>> _getJson(String path, [Map<String, String>? params]) async {
    final uri = Uri.parse('$baseUrl$path').replace(queryParameters: params);
    final resp = await _client.get(uri);
    final body = jsonDecode(utf8.decode(resp.bodyBytes)) as Map<String, dynamic>;
    if (resp.statusCode == 409) {
      throw ApiHintException(body['detail'] ?? 'invalid input', body['hint'] ?? '');
    }
    if (resp.statusCode != 200) {
      throw Exception('HTTP ${resp.statusCode} from $path: $body');
    }
    return body;
  }

  Future<TokenizeResult> tokenize(String text) async =>
      TokenizeResult.fromJson(await _getJson('/api/tokenize', {'text': text}));

  Future<EmbedResult> embed(String text) async =>
      EmbedResult.fromJson(await _getJson('/api/embed', {'text': text}));

  Future<NeighboursResult> neighbours(String text, {int k = 5}) async => NeighboursResult.fromJson(
        await _getJson('/api/neighbours', {'text': text, 'k': '$k'}),
      );

  Future<ClassifyResult> classify(String text) async =>
      ClassifyResult.fromJson(await _getJson('/api/classify', {'text': text}));

  Future<MapResult> map() async => MapResult.fromJson(await _getJson('/api/map'));

  Future<DemosResult> demos() async => DemosResult.fromJson(await _getJson('/api/demos'));
}
