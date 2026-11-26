import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

torch = pytest.importorskip("torch")

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from stylelens.embeddings import EmbeddingNet, extract  # noqa: E402

# Poids aléatoires : pas de téléchargement, résultats déterministes une fois le modèle construit.
SETTINGS = Settings(model_name="mobilenet_v3_large", model_pretrained=False, image_size=64, device="cpu")


def png(color=(200, 30, 30), size=(60, 80), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app(SETTINGS)) as c:
        yield c


def test_health_describes_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body == {"status": "UP", "model": "mobilenet_v3_large", "fine_tuned": False, "dim": 960,
                    "image_size": 64, "device": "cpu"}


def test_embed_returns_normalized_vector(client):
    r = client.post("/embed", files={"file": ("red.png", png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dim"] == len(body["embedding"]) == 960
    assert body["model"] == "mobilenet_v3_large" and body["fine_tuned"] is False
    assert np.linalg.norm(body["embedding"]) == pytest.approx(1.0, abs=1e-5)


def test_embed_matches_offline_catalog_extraction(client, tmp_path):
    """Même vecteur que stylelens.embeddings.extract (pipeline qui a indexé le catalogue)."""
    path = tmp_path / "item.png"
    path.write_bytes(png((10, 120, 200)))
    online = client.post("/embed", files={"file": ("item.png", path.read_bytes(), "image/png")}).json()
    offline = extract(client.app.state.model.model, [path], image_size=64, device="cpu")[0]
    np.testing.assert_allclose(online["embedding"], offline, atol=1e-5)


@pytest.mark.parametrize(("payload", "content_type", "status"), [
    (b"pas une image", "image/png", 400),
    (b"", "image/png", 400),
    (b"%PDF-1.4", "application/pdf", 415),
])
def test_embed_rejects_invalid_files(client, payload, content_type, status):
    r = client.post("/embed", files={"file": ("f", payload, content_type)})
    assert r.status_code == status
    assert r.json()["detail"]


def test_embed_requires_a_file(client):
    assert client.post("/embed").status_code == 422


def test_embed_rejects_oversized_upload():
    settings = Settings(**{**SETTINGS.__dict__, "max_upload_bytes": 1000})
    noisy = io.BytesIO()  # bruit : PNG incompressible, bien au-delà de 1 000 octets
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)).save(noisy, "PNG")
    with TestClient(create_app(settings)) as c:
        r = c.post("/embed", files={"file": ("big.png", noisy.getvalue(), "image/png")})
        assert r.status_code == 413
        assert c.post("/embed", files={"file": ("ok.png", png(size=(8, 8)), "image/png")}).status_code == 200


def test_fine_tuned_checkpoint_is_served(tmp_path):
    model = EmbeddingNet("mobilenet_v3_large", embed_dim=16, pretrained=False).eval()
    ckpt = tmp_path / "model.pt"
    torch.save({"backbone": "mobilenet_v3_large", "embed_dim": 16, "image_size": 96, "epoch": 3,
                "state_dict": model.state_dict()}, ckpt)
    settings = Settings(model_checkpoint=ckpt, model_pretrained=False, device="cpu")
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").json() | {"device": "cpu"} == {
            "status": "UP", "model": "mobilenet_v3_large", "fine_tuned": True, "dim": 16,
            "image_size": 96, "device": "cpu"}
        r = c.post("/embed", files={"file": ("x.png", png(), "image/png")})
        assert r.json()["dim"] == 16


def test_missing_checkpoint_fails_at_startup(tmp_path):
    settings = Settings(model_checkpoint=tmp_path / "absent.pt", device="cpu")
    with pytest.raises(FileNotFoundError, match="MODEL_CHECKPOINT"):
        with TestClient(create_app(settings)):
            pass


def test_settings_from_env():
    s = Settings.from_env({"MODEL_CHECKPOINT": "/models/m.pt", "MODEL_PRETRAINED": "false",
                           "IMAGE_SIZE": "128", "MAX_UPLOAD_BYTES": "2048"})
    assert str(s.model_checkpoint).replace("\\", "/") == "/models/m.pt"
    assert s.model_pretrained is False and s.image_size == 128 and s.max_upload_bytes == 2048
    assert Settings.from_env({}).model_checkpoint is None
