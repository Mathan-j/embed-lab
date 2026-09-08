import 'package:flutter/material.dart';

/// One fixed colour per category, used everywhere a category needs a colour:
/// neighbour chips, the prediction bars, and the cluster map legend. Kept in
/// one place so a category never gets two different colours on screen.
const Map<String, Color> kCategoryColors = {
  'animal': Color(0xFF2E7D32),
  'food': Color(0xFFEF6C00),
  'vehicle': Color(0xFF1565C0),
  'emotion': Color(0xFFAD1457),
  'colour': Color(0xFF6A1B9A),
  'job': Color(0xFF00838F),
};

Color colorForCategory(String category) =>
    kCategoryColors[category] ?? const Color(0xFF616161);
