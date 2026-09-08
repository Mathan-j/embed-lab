# Embed Lab -- Flutter frontend

One screen, five cards: type a word, see its tokens, its 384-dim embedding, its
nearest neighbours, what a trained classifier thinks it is, and where it lands
on a 600-word cluster map. Every number comes from the live backend -- nothing
in this app is hardcoded, including the two "try these" demo lists.

## Backend

Start the backend first, on port 8100 (8000 is taken by an unrelated service
on this machine):

```bash
cd ../backend
uv run uvicorn app.main:app --port 8100
```

## Base URL resolution (`lib/api.dart`)

1. `--dart-define=API_BASE=...` always wins if passed.
2. Otherwise, on **Android** the default is `http://10.0.2.2:8100` -- the
   Android emulator's alias for the host machine's `localhost`. Plain
   `localhost` inside the emulator means the emulator itself, so this has to
   be handled explicitly or the app silently fails to reach the backend.
3. Every other platform (web, Windows, desktop Chrome/Edge) defaults to
   `http://localhost:8100`.

## Run commands

**Web (Chrome), backend on 8100, default base URL:**

```bash
flutter run -d chrome
```

**Windows desktop:**

```bash
flutter run -d windows
```

**Android emulator/device** (needs the explicit `10.0.2.2` default, already
built in -- no `--dart-define` required unless you want to override it):

```bash
flutter run -d <android-device-id>
```

**Overriding the backend location explicitly** (e.g. backend on a different
host/port, or testing the Android path from a desktop target):

```bash
flutter run -d chrome --dart-define=API_BASE=http://localhost:8100
flutter run -d windows --dart-define=API_BASE=http://192.168.1.20:8100
```

## Builds

```bash
flutter build web
flutter build apk --debug
flutter build windows
```

## CORS

The backend's `CORSMiddleware` (`backend/app/main.py`) allows any
`http://localhost:<port>` origin via `allow_origin_regex` -- permissive on
purpose, since this is a localhost-only teaching app with no deployment
target. **Do not ship this CORS policy as-is to anything with a real
origin/deployment.**

## What is deliberately not here

No routing, no state-management package, no theming system, no charting
library -- the sparkline and cluster map are both `CustomPainter`. This is an
MVP: `setState` is enough for one screen.

## A deliberate honesty gap: the cluster map only plots the typed word if it
is one of the 600 mapped words

The backend does not expose a way to project an arbitrary typed word into the
map's 2-D PCA space (that would need the same fitted `PCA` transform, which
`/api/map` does not return). Rather than fake a plausible-looking position,
`MapCard` only highlights the typed word when it exactly matches one of the
600 words already in `/api/map`'s `words` list, and says so plainly otherwise.
