// This app makes real network calls to the embed-lab backend on startup
// (map, demos, the contrast pair, and the default "cat" query), so a full
// pumpWidget smoke test would need a live backend or an injected client.
//
// Note what does NOT work here: pointing the API at a dead port. `flutter test`
// installs an HttpOverrides that answers every request with a canned 400 and
// never opens a socket, so no real ClientException is ever raised -- the
// framework says as much in its own failure output. The unreachable-backend
// path therefore has to be driven by injecting a client that throws, which is
// what `_ThrowingClient` below does.

import 'package:embed_lab/api.dart';
import 'package:embed_lab/main.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;

const _deadBackend = 'http://localhost:1';

/// Raises the exact exception package:http raises when the host cannot be
/// reached: on web a "Failed to fetch", on native a wrapped SocketException.
class _ThrowingClient extends http.BaseClient {
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    throw http.ClientException('Failed to fetch', request.url);
  }
}

/// A client that answers every endpoint with a 409 + hint, to prove the banner
/// logic does not swallow a real API error the user is meant to read.
class _HintClient extends http.BaseClient {
  @override
  Future<http.StreamedResponse> send(http.BaseRequest request) async {
    return http.StreamedResponse(
      Stream.value(utf8Bytes('{"detail":"text is required","hint":"try cat"}')),
      409,
      request: request,
    );
  }
}

List<int> utf8Bytes(String s) => s.codeUnits;

Future<void> _pump(WidgetTester tester, http.Client client) async {
  // The default test viewport is 800x600 and the card list builds lazily, so
  // the lower cards are never laid out and `find.text('Prediction')` finds
  // nothing. A tall surface makes all five present at once, which is what
  // these tests are asserting about.
  await tester.binding.setSurfaceSize(const Size(1200, 3000));
  addTearDown(() => tester.binding.setSurfaceSize(null));

  await tester.pumpWidget(
    MaterialApp(
      home: EmbedLabScreen(api: EmbedLabApi(baseUrl: _deadBackend, client: client)),
    ),
  );
  await tester.pump(const Duration(seconds: 1));
  await tester.pump(const Duration(seconds: 1));
}

void main() {
  test('resolveBaseUrl defaults to localhost:8100 outside Android', () {
    // On the test VM (and web/Windows/desktop at runtime) kIsWeb is false
    // and Platform.isAndroid is false, so this exercises the non-Android
    // branch of api.dart's resolveBaseUrl().
    expect(resolveBaseUrl(), 'http://localhost:8100');
  });

  testWidgets('an unreachable backend shows one banner naming the URL',
      (tester) async {
    await _pump(tester, _ThrowingClient());

    // The banner must say what is wrong and what to do, and name the URL it
    // tried -- a visitor who has never seen this project has to be able to act
    // on it.
    expect(find.textContaining("Can't reach the API"), findsOneWidget);
    expect(find.textContaining(_deadBackend), findsOneWidget);
    expect(find.byIcon(Icons.cloud_off), findsOneWidget);
  });

  testWidgets('the banner replaces the per-card errors rather than joining them',
      (tester) async {
    await _pump(tester, _ThrowingClient());

    // The defect this guards: five cards each rendering their own raw
    // "Could not load: ClientException: Failed to fetch, uri=..." is what a
    // visitor to the published build used to see, and it reads as a broken app
    // rather than a missing backend. One explanation, not five stack traces.
    expect(find.textContaining('Could not load'), findsNothing);
    expect(find.textContaining('ClientException'), findsNothing);

    // Nor may anything still claim to be loading. Two elements saying
    // different things about the same failure ("can't reach the API" above
    // "Loading the 600-word map...") reads as a half-broken app, and those
    // futures have already rejected -- they will never resolve.
    expect(find.textContaining('Loading'), findsNothing);
    expect(find.textContaining('loading contrast pair'), findsNothing);

    // ...while the cards themselves are still on screen, in a neutral state.
    for (final title in [
      'Tokens',
      'Embedding',
      'Neighbours',
      'Prediction',
      'Cluster map',
    ]) {
      expect(find.text(title), findsOneWidget, reason: '$title card missing');
    }
  });

  testWidgets('a real 409 still shows its hint and raises no banner',
      (tester) async {
    // The distinction the fix turns on: a 409 is the backend ANSWERING with
    // something the user can act on, so its hint must reach the screen
    // verbatim. Only an unreachable host gets the banner. If the banner logic
    // keyed on "any error" instead of ClientException, this test fails.
    await _pump(tester, _HintClient());

    expect(find.textContaining("Can't reach the API"), findsNothing);
    expect(find.byIcon(Icons.cloud_off), findsNothing);
    expect(find.textContaining('try cat'), findsWidgets);
  });
}
