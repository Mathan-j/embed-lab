import 'package:flutter/material.dart';

/// Scatter-plots the 600-word PCA projection from /api/map, coloured by
/// cluster (not by true category -- the point of the map is to show what an
/// *unsupervised* method found on its own). `CustomPainter`, not a charting
/// package, per the MVP rule of one screen / no new dependencies.
class ClusterMapPainter extends CustomPainter {
  final List<List<double>> coords; // 600 x 2, raw PCA units
  final List<int> clusters; // 600, cluster index 0..n-1
  final List<Color> clusterColors;
  final int? highlightIndex; // index into coords/clusters, or null

  ClusterMapPainter({
    required this.coords,
    required this.clusters,
    required this.clusterColors,
    this.highlightIndex,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (coords.isEmpty) return;

    double minX = coords[0][0], maxX = coords[0][0];
    double minY = coords[0][1], maxY = coords[0][1];
    for (final p in coords) {
      if (p[0] < minX) minX = p[0];
      if (p[0] > maxX) maxX = p[0];
      if (p[1] < minY) minY = p[1];
      if (p[1] > maxY) maxY = p[1];
    }
    final rangeX = (maxX - minX).abs() < 1e-9 ? 1.0 : (maxX - minX);
    final rangeY = (maxY - minY).abs() < 1e-9 ? 1.0 : (maxY - minY);

    const padding = 12.0;
    Offset project(List<double> p) {
      final nx = (p[0] - minX) / rangeX;
      final ny = (p[1] - minY) / rangeY;
      final x = padding + nx * (size.width - 2 * padding);
      // Flip Y: screen-down is positive, PCA axis 2 conventionally points up.
      final y = size.height - (padding + ny * (size.height - 2 * padding));
      return Offset(x, y);
    }

    final dotPaint = Paint()..style = PaintingStyle.fill;
    for (var i = 0; i < coords.length; i++) {
      if (i == highlightIndex) continue; // drawn last, on top
      final c = clusterColors[clusters[i] % clusterColors.length];
      dotPaint.color = c.withValues(alpha: 0.55);
      canvas.drawCircle(project(coords[i]), 2.2, dotPaint);
    }

    if (highlightIndex != null && highlightIndex! < coords.length) {
      final center = project(coords[highlightIndex!]);
      final ring = Paint()
        ..color = Colors.black
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.5;
      canvas.drawCircle(center, 7, ring);
      final fill = Paint()
        ..color = clusterColors[clusters[highlightIndex!] % clusterColors.length];
      canvas.drawCircle(center, 5, fill);
    }
  }

  @override
  bool shouldRepaint(covariant ClusterMapPainter oldDelegate) =>
      oldDelegate.coords != coords ||
      oldDelegate.clusters != clusters ||
      oldDelegate.highlightIndex != highlightIndex;
}

/// A fixed, distinguishable palette for cluster indices (up to n_clusters=6).
/// Deliberately separate from `kCategoryColors`: this palette labels
/// *clusters* (what KMeans found), not *categories* (the true label), and
/// the map's honesty depends on that distinction staying visible -- if
/// clusters happened to reuse category colours it would look like the
/// clustering is shown to already know the categories.
const List<Color> kClusterPalette = [
  Color(0xFFD32F2F),
  Color(0xFF1976D2),
  Color(0xFF388E3C),
  Color(0xFFF57F17),
  Color(0xFF7B1FA2),
  Color(0xFF00838F),
  Color(0xFF5D4037),
  Color(0xFFC2185B),
];
