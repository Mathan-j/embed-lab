import 'package:flutter/material.dart';
import '../api.dart';
import '../painters/cluster_map_painter.dart';
import 'card_shell.dart';

/// Card 5: the 600-word PCA scatter, coloured by the *unsupervised* cluster
/// KMeans found (not the true category), with the typed word highlighted
/// when it is one of the 600 mapped words.
///
/// Honesty note: the backend has no endpoint to project an arbitrary typed
/// word into this 2-D space (it would need the same fitted PCA transform,
/// which is not exposed). Rather than fake a position, the typed word is
/// only plotted when it is literally one of the 600 words already in the
/// map; otherwise the card says so plainly instead of guessing a spot.
class MapCard extends StatelessWidget {
  final MapResult? data;
  final bool loading;
  final Object? error;
  final String typedWord;

  const MapCard({
    super.key,
    required this.data,
    required this.loading,
    required this.error,
    required this.typedWord,
  });

  @override
  Widget build(BuildContext context) {
    int? highlightIndex;
    if (data != null && typedWord.trim().isNotEmpty) {
      final lower = typedWord.trim().toLowerCase();
      final idx = data!.words.indexWhere((w) => w.toLowerCase() == lower);
      if (idx != -1) highlightIndex = idx;
    }

    return CardShell(
      title: 'Cluster map',
      loading: loading,
      error: error,
      child: data == null
          ? const Text('Loading the 600-word map...')
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'ARI: ${data!.ari.toStringAsFixed(4)}  (${data!.nClusters} clusters, seed ${data!.seed})',
                  style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
                ),
                const Text(
                  'ARI is already chance-corrected: 0.0 means the clustering is no '
                  'better than random, so this number needs no separate baseline.',
                  style: TextStyle(fontSize: 11, color: Colors.grey),
                ),
                const SizedBox(height: 8),
                SizedBox(
                  height: 260,
                  width: double.infinity,
                  child: CustomPaint(
                    painter: ClusterMapPainter(
                      coords: data!.coords,
                      clusters: data!.clusters,
                      clusterColors: kClusterPalette,
                      highlightIndex: highlightIndex,
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                if (typedWord.trim().isEmpty)
                  const Text('Type a word to see if it is one of the 600 mapped words.',
                      style: TextStyle(fontSize: 11, color: Colors.grey))
                else if (highlightIndex == null)
                  Text(
                    '"$typedWord" is not one of the 600 mapped words, so it is not '
                    'plotted -- the backend has no way to place an arbitrary word on '
                    'this map without re-fitting the projection.',
                    style: const TextStyle(fontSize: 11, color: Colors.grey),
                  )
                else
                  Text(
                    '"$typedWord" is plotted (ringed dot) -- cluster '
                    '${data!.clusters[highlightIndex]}, true category '
                    '${data!.categories[highlightIndex]}.',
                    style: const TextStyle(fontSize: 11, color: Colors.black87),
                  ),
              ],
            ),
    );
  }
}
