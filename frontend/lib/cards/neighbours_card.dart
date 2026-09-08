import 'package:flutter/material.dart';
import '../api.dart';
import '../colors.dart';
import 'card_shell.dart';

/// Card 3: top-5 nearest neighbours by cosine similarity, plus a fixed
/// contrast row -- cos(cat, dog) vs cos(cat, car) -- so "spelling is not
/// meaning" is visible on screen, not just a fact about the dataset.
/// `catDogCosine`/`catCarCosine` are fetched live from /api/embed by the
/// caller (never hardcoded), even though the values are stable.
class NeighboursCard extends StatelessWidget {
  final NeighboursResult? data;
  final bool loading;
  final Object? error;
  final double? catDogCosine;
  final double? catCarCosine;

  const NeighboursCard({
    super.key,
    required this.data,
    required this.loading,
    required this.error,
    required this.catDogCosine,
    required this.catCarCosine,
  });

  @override
  Widget build(BuildContext context) {
    return CardShell(
      title: 'Neighbours',
      loading: loading,
      error: error,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (data == null)
            const Text('Type something above.')
          else
            ...data!.hits.map((h) => Padding(
                  padding: const EdgeInsets.symmetric(vertical: 3),
                  child: Row(
                    children: [
                      Container(
                        width: 10,
                        height: 10,
                        decoration: BoxDecoration(
                          color: colorForCategory(h.category),
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(child: Text('${h.word}  (${h.category})')),
                      Text(h.score.toStringAsFixed(4)),
                    ],
                  ),
                )),
          const Divider(height: 20),
          Text(
            'Spelling is not meaning -- one letter apart is not one meaning apart:',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 4),
          if (catDogCosine == null || catCarCosine == null)
            const Text('loading contrast pair...', style: TextStyle(fontSize: 12))
          else
            Text(
              'cos(cat, dog) = ${catDogCosine!.toStringAsFixed(4)}   '
              'cos(cat, car) = ${catCarCosine!.toStringAsFixed(4)}',
              style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
            ),
        ],
      ),
    );
  }
}
