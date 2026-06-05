"""Seletor de backend (VPS / OpenRouter / Auto) na tela de Admin.

Testa get/set do backend_mode (SQLite) e que resolve_model_alias respeita o
modo: 'openrouter' manda os aliases pro modelo grátis; 'vps'/'auto' usam a VPS.
Stuba rotator_library.anthropic_compat pra importar model_resolution sem litellm.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def db(monkeypatch):
    """admin_db apontado pra um SQLite temporário (isolado por teste)."""
    monkeypatch.setenv("ADMIN_DB_PATH", tempfile.mktemp(suffix=".db"))
    # garante reimport limpo do módulo com o novo path
    sys.modules.pop("proxy_app.admin_db", None)
    from proxy_app import admin_db
    return admin_db


@pytest.fixture
def mr(monkeypatch):
    """model_resolution importável sem litellm (stuba o anthropic_compat)."""
    if "rotator_library" not in sys.modules:
        monkeypatch.setitem(sys.modules, "rotator_library", types.ModuleType("rotator_library"))
    ac = types.ModuleType("rotator_library.anthropic_compat")
    ac.AnthropicMessagesRequest = object
    monkeypatch.setitem(sys.modules, "rotator_library.anthropic_compat", ac)
    monkeypatch.setenv("HOSTED_VLLM_API_BASE", "https://fake-pod/v1")
    monkeypatch.setenv("PROXY_DEFAULT_MODEL", "hosted_vllm/qwen2.5-coder-32b")
    sys.modules.pop("proxy_app.model_resolution", None)
    from proxy_app import model_resolution
    return model_resolution


def test_backend_mode_default_is_vps(db):
    assert db.get_backend_mode() == "vps"


def test_backend_mode_set_get(db):
    assert db.set_backend_mode("openrouter") == "openrouter"
    assert db.get_backend_mode() == "openrouter"
    assert db.set_backend_mode("auto") == "auto"
    assert db.get_backend_mode() == "auto"


def test_backend_mode_invalid_falls_to_vps(db):
    db.set_backend_mode("lixo-invalido")
    assert db.get_backend_mode() == "vps"


def test_openrouter_fallback_model_default(mr):
    assert mr.openrouter_fallback_model() == "openrouter/qwen/qwen3-coder:free"


def test_openrouter_fallback_model_override(mr, monkeypatch):
    monkeypatch.setenv("OPENROUTER_FALLBACK_MODEL", "openrouter/algum/pago")
    assert mr.openrouter_fallback_model() == "openrouter/algum/pago"


def test_resolve_vps_mode_uses_vps(db, mr, monkeypatch):
    monkeypatch.setattr(mr, "_current_backend_mode", lambda: "vps")
    assert mr.resolve_model_alias("claude-sonnet-4-5") == "hosted_vllm/qwen2.5-coder-32b"


def test_resolve_openrouter_mode_uses_openrouter(db, mr, monkeypatch):
    monkeypatch.setattr(mr, "_current_backend_mode", lambda: "openrouter")
    assert mr.resolve_model_alias("claude-sonnet-4-5") == "openrouter/qwen/qwen3-coder:free"


def test_resolve_auto_mode_uses_vps(db, mr, monkeypatch):
    # 'auto' usa a VPS aqui; o fallback é na rota /v1/messages, não no resolve
    monkeypatch.setattr(mr, "_current_backend_mode", lambda: "auto")
    assert mr.resolve_model_alias("claude-sonnet-4-5") == "hosted_vllm/qwen2.5-coder-32b"


def test_resolve_explicit_model_unchanged(mr, monkeypatch):
    # modelo que já tem provider/ não é tocado, qualquer que seja o modo
    monkeypatch.setattr(mr, "_current_backend_mode", lambda: "openrouter")
    assert mr.resolve_model_alias("openrouter/x/y") == "openrouter/x/y"
