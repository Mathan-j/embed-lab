"""End-to-end checks through the actual HTTP surface (TestClient, no real server,
no network) -- the app.vocab / app.tokenize / app.embed unit tests already cover
the semantics in detail, so these confirm the wiring: routes exist, responses have
the right shape, and user-fixable errors come back as 409 with detail+hint rather
than a stack trace.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan, building the VocabIndex once
        yield c


def test_tokenize_endpoint_returns_pieces_ids_and_count(client):
    r = client.get("/api/tokenize", params={"text": "cat"})
    assert r.status_code == 200
    body = r.json()
    assert body["pieces"] == ["cat"]
    assert body["ids"] == [body["ids"][0]]
    assert body["count"] == 1


def test_tokenize_endpoint_rejects_empty_text_with_409_and_hint(client):
    r = client.get("/api/tokenize", params={"text": ""})
    assert r.status_code == 409
    body = r.json()
    assert "detail" in body and "hint" in body


def test_embed_endpoint_returns_384_dims_and_unit_norm(client):
    r = client.get("/api/embed", params={"text": "cat"})
    assert r.status_code == 200
    body = r.json()
    assert body["dims"] == 384
    assert len(body["vector"]) == 384
    assert abs(body["norm"] - 1.0) < 1e-6


def test_embed_endpoint_rejects_empty_text_with_409(client):
    r = client.get("/api/embed", params={"text": "   "})
    assert r.status_code == 409


def test_neighbours_endpoint_returns_k_hits_excluding_the_query_word(client):
    r = client.get("/api/neighbours", params={"text": "cat", "k": 5})
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert len(hits) == 5
    assert all(h["word"] != "cat" for h in hits)
    animals = [h for h in hits if h["category"] == "animal"]
    assert len(animals) >= 3, f"expected mostly animals, got {hits}"


def test_neighbours_endpoint_rejects_k_out_of_range_with_409(client):
    r = client.get("/api/neighbours", params={"text": "cat", "k": 0})
    assert r.status_code == 409
    body = r.json()
    assert "detail" in body and "hint" in body

    r = client.get("/api/neighbours", params={"text": "cat", "k": 10_000})
    assert r.status_code == 409


def test_neighbours_endpoint_rejects_empty_text_with_409(client):
    r = client.get("/api/neighbours", params={"text": ""})
    assert r.status_code == 409
