import 'package:flutter/material.dart';
import '../api.dart';

/// Common chrome for all five cards: a title, and one of three states --
/// loading, a readable error (the backend's 409 `hint`, verbatim), or the
/// built child once data has arrived.
class CardShell extends StatelessWidget {
  final String title;
  final bool loading;
  final Object? error;
  final Widget child;

  const CardShell({
    super.key,
    required this.title,
    required this.loading,
    required this.error,
    required this.child,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            if (loading)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 16),
                child: Center(child: CircularProgressIndicator()),
              )
            else if (error != null)
              _ErrorMessage(error: error!)
            else
              child,
          ],
        ),
      ),
    );
  }
}

class _ErrorMessage extends StatelessWidget {
  final Object error;
  const _ErrorMessage({required this.error});

  @override
  Widget build(BuildContext context) {
    // A 409 carries a human-readable hint -- show that verbatim, since the
    // whole point of the backend producing one is that it beats a generic
    // "something went wrong" message.
    final text = error is ApiHintException
        ? '${(error as ApiHintException).detail}\n${(error as ApiHintException).hint}'
        : 'Could not load: $error';
    return Text(text, style: const TextStyle(color: Colors.red));
  }
}
