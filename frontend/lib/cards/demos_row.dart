import 'package:flutter/material.dart';
import '../api.dart';

/// The "try these" row, built from /api/demos rather than hardcoded in Dart
/// -- these two lists (AMBIGUOUS, SUBWORD_LEAKAGE) are the two most
/// interesting things in the app and must stay sourced from the backend's
/// vocabulary, not copied into the client.
class DemosRow extends StatelessWidget {
  final DemosResult? data;
  final bool loading;
  final Object? error;
  final void Function(String word) onPick;

  const DemosRow({
    super.key,
    required this.data,
    required this.loading,
    required this.error,
    required this.onPick,
  });

  @override
  Widget build(BuildContext context) {
    if (loading) {
      return const Padding(
        padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        child: LinearProgressIndicator(minHeight: 2),
      );
    }
    if (error != null || data == null) {
      return const SizedBox.shrink();
    }
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Try these -- ambiguous words (a confident answer is not a bug; '
            'the label genuinely fits two categories):',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 4),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: data!.ambiguous
                .map((w) => ActionChip(label: Text(w), onPressed: () => onPick(w)))
                .toList(),
          ),
          const SizedBox(height: 10),
          Text(
            'Try these -- subword-leakage pairs (the compound shares a wordpiece '
            'with the plain word, so its neighbour score to it is inflated by '
            'spelling, not confused meaning):',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 4),
          Wrap(
            spacing: 6,
            runSpacing: 6,
            children: data!.subwordLeakage.expand((p) => [
                  ActionChip(label: Text(p.plain), onPressed: () => onPick(p.plain)),
                  ActionChip(label: Text(p.compound), onPressed: () => onPick(p.compound)),
                ]).toList(),
          ),
        ],
      ),
    );
  }
}
