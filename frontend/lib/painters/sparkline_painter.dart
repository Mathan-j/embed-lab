import 'package:flutter/material.dart';

/// Draws all 384 embedding values as a simple line sparkline. Custom-painted
/// rather than pulled from a charting package -- an MVP scatter/line plot
/// does not need one, per the project's "no dependency beyond http" rule.
class SparklinePainter extends CustomPainter {
  final List<double> values;
  final Color color;
  SparklinePainter(this.values, {this.color = Colors.blueGrey});

  @override
  void paint(Canvas canvas, Size size) {
    if (values.isEmpty) return;
    final minV = values.reduce((a, b) => a < b ? a : b);
    final maxV = values.reduce((a, b) => a > b ? a : b);
    final range = (maxV - minV).abs() < 1e-9 ? 1.0 : (maxV - minV);

    final path = Path();
    final dx = size.width / (values.length - 1).clamp(1, values.length);
    for (var i = 0; i < values.length; i++) {
      final x = i * dx;
      final norm = (values[i] - minV) / range; // 0..1
      final y = size.height - norm * size.height;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final paint = Paint()
      ..color = color
      ..strokeWidth = 1.2
      ..style = PaintingStyle.stroke;
    canvas.drawPath(path, paint);

    // A faint zero line for reference, since values are roughly centred on 0.
    final zeroNorm = (0 - minV) / range;
    final zeroY = size.height - zeroNorm * size.height;
    if (zeroY >= 0 && zeroY <= size.height) {
      final zeroPaint = Paint()
        ..color = color.withValues(alpha: 0.25)
        ..strokeWidth = 1;
      canvas.drawLine(Offset(0, zeroY), Offset(size.width, zeroY), zeroPaint);
    }
  }

  @override
  bool shouldRepaint(covariant SparklinePainter oldDelegate) =>
      oldDelegate.values != values || oldDelegate.color != color;
}
