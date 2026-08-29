import io
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module

FAKE_MANIFEST = {
    "selected_backbone": "EfficientNetB0",
    "stage": "finetuned",
    "val_auc_macro": 0.95,
    "test_auc_macro": 0.93,
    "num_classes": 3,
    "class_names": ["aphid", "beetle", "grasshopper"],
}


class FakeModel:
    def predict(self, x, verbose=0):
        return np.array([[0.1, 5.0, 0.2]])  # "beetle" wins


def _make_image_bytes(fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def client(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(FAKE_MANIFEST))
    monkeypatch.setattr(app_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(app_module, "load_saved_model", lambda backbone, stage: FakeModel())

    with TestClient(app_module.app) as c:
        yield c


# ---------- /health ----------

def test_health_reports_model_loaded(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model_loaded": True}


def test_health_returns_503_when_model_load_fails(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(FAKE_MANIFEST))
    monkeypatch.setattr(app_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        app_module, "load_saved_model",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with TestClient(app_module.app) as c:
        resp = c.get("/health")
        assert resp.status_code == 503
        assert resp.json() == {"detail": "Model not loaded"}


# ---------- /predict ----------

def test_predict_success(client):
    files = {"file": ("bug.png", _make_image_bytes(), "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "beetle"
    assert len(body["top_3"]) == 3


def test_predict_rejects_unsupported_content_type(client):
    files = {"file": ("bug.txt", b"not an image", "text/plain")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 415


def test_predict_rejects_oversized_file(client):
    big_bytes = b"0" * (6 * 1024 * 1024)
    files = {"file": ("bug.png", big_bytes, "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 413


def test_predict_rejects_corrupt_image(client):
    files = {"file": ("bug.png", b"not really a png", "image/png")}
    resp = client.post("/predict", files=files)
    assert resp.status_code == 400


def test_predict_returns_503_when_model_not_loaded(monkeypatch, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(FAKE_MANIFEST))
    monkeypatch.setattr(app_module, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        app_module, "load_saved_model",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with TestClient(app_module.app) as c:
        files = {"file": ("bug.png", _make_image_bytes(), "image/png")}
        resp = c.post("/predict", files=files)
        assert resp.status_code == 503


# ---------- /model/info ----------

def test_model_info_returns_manifest_fields(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["selected_backbone"] == "EfficientNetB0"
    assert body["num_classes"] == 3
    assert body["class_names"] == ["aphid", "beetle", "grasshopper"]
    assert body["test_auc_macro"] == 0.93