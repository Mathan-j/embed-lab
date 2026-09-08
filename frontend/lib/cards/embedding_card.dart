import 'package:flutter/material.dart';
import '../api.dart';
import '../painters/sparkline_painter.dart';
import 'card_shell.dart';

/// Card 2: dims, norm, the first 12 values, and a sparkline of all 384.
class EmbeddingCard extends StatelessWidget {
  final EmbedResult? data;
  final bool loading;
  final Object? error;

  const EmbeddingCard({super.key, required this.data, required this.loading, required this.error});

  @override
  Widget build(BuildContext context) {
    return CardShell(
      title: 'Embedding',
      loading: loading,
      error: error,
      child: data == null
          ? const Text('Type something above.')
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('dims: ${data!.dims}   norm: ${data!.norm.toStringAsFixed(4)}'),
                const SizedBox(height: 8),
                Text(
                  'first 12: ${data!.vector.take(12).map((v) => v.toStringAsFixed(3)).join(', ')}',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
                const SizedBox(height: 10),
                SizedBox(
                  height: 60,
                  width: double.infinity,
                  child: CustomPaint(
                    painter: SparklinePainter(data!.vector, color: Colors.indigo),
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'all 384 values, in order (not sorted by magnitude)',
                  style: TextStyle(fontSize: 11, color: Colors.grey),
                ),
              ],
            ),
    );
  }
}
