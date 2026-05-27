"""Tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


class TestHealth:
    def test_root(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "TCM" in r.json()["name"]

    def test_health(self):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


class TestProtocolAPI:
    def test_list_empty(self):
        r = client.get("/api/protocols/")
        assert r.status_code == 200

    def test_create_and_get(self):
        payload = {"trial_id": "TEST-001", "title": "Test Protocol", "description": "Test"}
        r = client.post("/api/protocols/", json=payload)
        assert r.status_code == 200
        assert r.json()["trial_id"] == "TEST-001"
        r2 = client.get("/api/protocols/TEST-001")
        assert r2.status_code == 200

    def test_duplicate(self):
        p = {"trial_id": "DUP-001", "title": "Dup"}
        client.post("/api/protocols/", json=p)
        assert client.post("/api/protocols/", json=p).status_code == 409

    def test_not_found(self):
        assert client.get("/api/protocols/NOPE").status_code == 404


class TestAnalysisAPI:
    def test_sensitivity(self):
        r = client.post("/api/analysis/sensitivity", json={
            "estimate": 0.65, "se": 0.1, "ci_lower": 0.45, "ci_upper": 0.85, "analysis_type": "both"})
        assert r.status_code == 200
        d = r.json()
        assert "e_value" in d and "tipping_point" in d
