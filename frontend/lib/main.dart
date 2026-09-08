import 'package:flutter/material.dart';

import 'api.dart';
import 'cards/demos_row.dart';
import 'cards/embedding_card.dart';
import 'cards/map_card.dart';
import 'cards/neighbours_card.dart';
import 'cards/prediction_card.dart';
import 'cards/tokens_card.dart';

void main() {
  runApp(const EmbedLabApp());
}

class EmbedLabApp extends StatelessWidget {
  const EmbedLabApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Embed Lab',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const EmbedLabScreen(),
    );
  }
}

class EmbedLabScreen extends StatefulWidget {
  const EmbedLabScreen({super.key});

  @override
  State<EmbedLabScreen> createState() => _EmbedLabScreenState();
}

class _EmbedLabScreenState extends State<EmbedLabScreen> {
  late final EmbedLabApi _api = EmbedLabApi();
  final _controller = TextEditingController(text: 'cat');

  String _queriedWord = '';

  bool _loadingWord = false;
  TokenizeResult? _tokenize;
  Object? _tokenizeError;
  EmbedResult? _embed;
  Object? _embedError;
  NeighboursResult? _neighbours;
  Object? _neighboursError;
  ClassifyResult? _classify;
  Object? _classifyError;

  bool _loadingMap = true;
  MapResult? _map;
  Object? _mapError;

  bool _loadingDemos = true;
  DemosResult? _demos;
  Object? _demosError;

  // The fixed cat/dog/car contrast row on the Neighbours card. Fetched once
  // from live /api/embed calls (never hardcoded) since the numbers are the
  // whole point of the card: spelling is not meaning.
  double? _catDogCosine;
  double? _catCarCosine;

  @override
  void initState() {
    super.initState();
    _loadMap();
    _loadDemos();
    _loadContrastPair();
    _runQuery(_controller.text);
  }

  Future<void> _loadMap() async {
    setState(() {
      _loadingMap = true;
      _mapError = null;
    });
    try {
      final m = await _api.map();
      setState(() => _map = m);
    } catch (e) {
      setState(() => _mapError = e);
    } finally {
      setState(() => _loadingMap = false);
    }
  }

  Future<void> _loadDemos() async {
    setState(() {
      _loadingDemos = true;
      _demosError = null;
    });
    try {
      final d = await _api.demos();
      setState(() => _demos = d);
    } catch (e) {
      setState(() => _demosError = e);
    } finally {
      setState(() => _loadingDemos = false);
    }
  }

  Future<void> _loadContrastPair() async {
    try {
      final results = await Future.wait([
        _api.embed('cat'),
        _api.embed('dog'),
        _api.embed('car'),
      ]);
      final cat = results[0].vector, dog = results[1].vector, car = results[2].vector;
      double dot(List<double> a, List<double> b) {
        var s = 0.0;
        for (var i = 0; i < a.length; i++) {
          s += a[i] * b[i];
        }
        return s;
      }

      setState(() {
        // Vectors are unit-norm (norm == 1.0 per /api/embed), so the dot
        // product is exactly the cosine similarity -- no extra division.
        _catDogCosine = dot(cat, dog);
        _catCarCosine = dot(cat, car);
      });
    } catch (_) {
      // The contrast row is illustrative, not load-bearing; a failure here
      // shows as "loading..." rather than blocking the rest of the screen.
    }
  }

  Future<void> _runQuery(String rawText) async {
    final text = rawText.trim();
    if (text.isEmpty) return;

    setState(() {
      _queriedWord = text;
      _loadingWord = true;
      _tokenizeError = null;
      _embedError = null;
      _neighboursError = null;
      _classifyError = null;
    });

    await Future.wait([
      _api.tokenize(text).then((v) => setState(() => _tokenize = v)).catchError(
          (e) => setState(() => _tokenizeError = e)),
      _api.embed(text).then((v) => setState(() => _embed = v)).catchError(
          (e) => setState(() => _embedError = e)),
      _api.neighbours(text, k: 5).then((v) => setState(() => _neighbours = v)).catchError(
          (e) => setState(() => _neighboursError = e)),
      _api.classify(text).then((v) => setState(() => _classify = v)).catchError(
          (e) => setState(() => _classifyError = e)),
    ]);

    if (mounted) setState(() => _loadingWord = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Embed Lab')),
      body: SafeArea(
        child: ListView(
          children: [
            Padding(
              padding: const EdgeInsets.all(16),
              child: TextField(
                controller: _controller,
                decoration: const InputDecoration(
                  labelText: 'Type a word or short phrase',
                  border: OutlineInputBorder(),
                  suffixIcon: Icon(Icons.search),
                ),
                onSubmitted: _runQuery,
              ),
            ),
            DemosRow(
              data: _demos,
              loading: _loadingDemos,
              error: _demosError,
              onPick: (word) {
                _controller.text = word;
                _runQuery(word);
              },
            ),
            const SizedBox(height: 4),
            TokensCard(data: _tokenize, loading: _loadingWord, error: _tokenizeError),
            EmbeddingCard(data: _embed, loading: _loadingWord, error: _embedError),
            NeighboursCard(
              data: _neighbours,
              loading: _loadingWord,
              error: _neighboursError,
              catDogCosine: _catDogCosine,
              catCarCosine: _catCarCosine,
            ),
            PredictionCard(data: _classify, loading: _loadingWord, error: _classifyError),
            MapCard(
              data: _map,
              loading: _loadingMap,
              error: _mapError,
              typedWord: _queriedWord,
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}
