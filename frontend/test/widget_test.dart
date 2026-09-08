// This app makes real network calls to the embed-lab backend on startup
// (map, demos, the contrast pair, and the default "cat" query), so a full
// pumpWidget smoke test would need a live backend or a mocked http client --
// out of scope for this MVP. This test instead pins the one thing that is
// pure and worth protecting: the Android-vs-everything-else base URL choice,
// which is exactly the kind of thing that silently breaks and is hard to
// notice (the emulator would just show "could not load" everywhere).

import 'package:flutter_test/flutter_test.dart';
import 'package:embed_lab/api.dart';

void main() {
  test('resolveBaseUrl defaults to localhost:8100 outside Android', () {
    // On the test VM (and web/Windows/desktop at runtime) kIsWeb is false
    // and Platform.isAndroid is false, so this exercises the non-Android
    // branch of api.dart's resolveBaseUrl().
    expect(resolveBaseUrl(), 'http://localhost:8100');
  });
}
