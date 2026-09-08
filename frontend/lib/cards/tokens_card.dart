import 'package:flutter/material.dart';
import '../api.dart';
import 'card_shell.dart';

/// Card 1: the wordpieces as chips with their ids, and the count. Subword
/// splits are the lesson here -- "vermilion" -> "ver ##mil ##ion" -- so the
/// chips are shown in order with the `##` continuation markers intact
/// rather than cleaned up.
class TokensCard extends StatelessWidget {
  final TokenizeResult? data;
  final bool loading;
  final Object? error;

  const TokensCard({super.key, required this.data, required this.loading, required this.error});

  @override
  Widget build(BuildContext context) {
    return CardShell(
      title: 'Tokens',
      loading: loading,
      error: error,
      child: data == null
          ? const Text('Type something above.')
          : Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: List.generate(data!.pieces.length, (i) {
                    return Chip(
                      label: Text('${data!.pieces[i]}  (${data!.ids[i]})'),
                    );
                  }),
                ),
                const SizedBox(height: 8),
                Text('${data!.count} wordpiece${data!.count == 1 ? '' : 's'}'),
              ],
            ),
    );
  }
}
