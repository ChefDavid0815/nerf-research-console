"""Desktop shell serves its bundled frontend from the local FastAPI origin."""

from fastapi.testclient import TestClient

from nerf_console.api import create_app


def test_desktop_frontend_and_api_share_an_origin(tmp_path):
    dist = tmp_path / "console_frontend" / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<title>NeRF desktop</title>", encoding="utf-8")
    (assets / "app.js").write_text("window.desktopReady = true", encoding="utf-8")

    client = TestClient(create_app(tmp_path))
    assert client.get("/").status_code == 200
    assert "NeRF desktop" in client.get("/").text
    assert client.get("/assets/app.js").text == "window.desktopReady = true"
    assert client.get("/api/health").json()["status"] == "ok"


def test_packaged_desktop_uses_its_own_offline_frontend(tmp_path, monkeypatch):
    packaged = tmp_path / "packaged-frontend"
    (packaged / "assets").mkdir(parents=True)
    (packaged / "index.html").write_text("<title>Packaged NeRF UI</title>", encoding="utf-8")
    (packaged / "assets" / "app.js").write_text("window.packaged = true", encoding="utf-8")
    monkeypatch.setenv("NERF_FRONTEND_DIST", str(packaged))

    client = TestClient(create_app(tmp_path / "workspace"))
    assert "Packaged NeRF UI" in client.get("/").text
    assert client.get("/assets/app.js").text == "window.packaged = true"
