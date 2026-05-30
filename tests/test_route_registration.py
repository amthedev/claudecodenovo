import sys
from pathlib import Path

from fastapi import FastAPI


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app.admin_routes import initialize_admin_db, register_admin_routes
from proxy_app.web_routes import register_web_routes


def _route_count(app: FastAPI, path: str) -> int:
    return sum(getattr(route, "path", None) == path for route in app.routes)


def test_admin_and_web_routes_are_registered_once():
    app = FastAPI()

    register_admin_routes(app, proxy_api_key="test-root")
    register_web_routes(app)
    register_admin_routes(app, proxy_api_key="test-root")
    register_web_routes(app)

    assert _route_count(app, "/admin") == 1
    assert _route_count(app, "/web") == 1
    assert _route_count(app, "/web/assets") == 1


def test_admin_db_initialization_is_explicit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    initialize_admin_db()

    assert (tmp_path / "admin.db").exists()
