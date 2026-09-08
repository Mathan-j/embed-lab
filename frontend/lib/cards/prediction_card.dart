import 'package:flutter/material.dart';
import '../api.dart';
import '../colors.dart';
import 'card_shell.dart';

/// Card 4: predicted category, a probability bar per category, and the
/// classifier's macro-F1 shown *with* its chance baseline. Never render
/// macro_f1 alone: chance on six balanced categories is 1/6 ~= 0.167, and a
/// bare 0.91 invites over-reading a number that only means something next
/// to what guessing would score.
class PredictionCard extends StatelessWidget {
  final ClassifyResult? data;
  final bool loading;
  final Object? error;

  const PredictionCard({super.key, required this.data, required this.loading, required this.error});

  @override
  Widget build(BuildContext context) {
    return CardShell(
      title: 'Prediction',
      loading: loading,
      error: error,
      child: data == null
          ? const Text('Type something above.')
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Text('predicted: ', style: TextStyle(fontWeight: FontWeight.bold)),
                    Text(
                      data!.predicted,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: colorForCategory(data!.predicted),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                ...(List<CategoryProbability>.from(data!.all)
                      ..sort((a, b) => b.probability.compareTo(a.probability)))
                    .map((p) => Padding(
                          padding: const EdgeInsets.symmetric(vertical: 3),
                          child: Row(
                            children: [
                              SizedBox(width: 70, child: Text(p.category)),
                              Expanded(
                                child: ClipRRect(
                                  borderRadius: BorderRadius.circular(3),
                                  child: LinearProgressIndicator(
                                    value: p.probability.clamp(0.0, 1.0),
                                    minHeight: 10,
                                    backgroundColor: Colors.grey.shade200,
                                    color: colorForCategory(p.category),
                                  ),
                                ),
                              ),
                              const SizedBox(width: 8),
                              SizedBox(
                                width: 50,
                                child: Text('${(p.probability * 100).toStringAsFixed(1)}%'),
                              ),
                            ],
                          ),
                        )),
                const Divider(height: 20),
                // The single most important discipline in this app: the F1
                // never appears without the baseline it has to beat.
                Text(
                  'macro-F1: ${data!.macroF1.toStringAsFixed(4)}   '
                  'vs. chance baseline: ${data!.baselineMacroF1.toStringAsFixed(4)} '
                  '(1 / 6 categories)',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
                const SizedBox(height: 2),
                const Text(
                  'measured on a held-out 30% test split, seed fixed for reproducibility',
                  style: TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
    );
  }
}
