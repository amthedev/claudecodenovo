"""Verifica que o snapshot da janela mostra clientes ativos MESMO com
USAGE_WINDOW=off — antes do fix o admin sempre via 0 porque check() retornava
antes de registrar."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _fresh_window():
    """Cria instância isolada (sem mexer no singleton do módulo)."""
    from proxy_app.usage_window import _UsageWindow
    return _UsageWindow()


def test_check_registers_client_even_when_disabled(monkeypatch):
    monkeypatch.setenv("USAGE_WINDOW", "off")
    w = _fresh_window()
    w.check("alice", 0.0)
    w.check("bob", 0.0)
    snap = w.snapshot()
    assert snap["enabled"] is False
    assert snap["active_clients"] == 2


def test_check_registers_client_when_enabled(monkeypatch):
    monkeypatch.setenv("USAGE_WINDOW", "on")
    w = _fresh_window()
    w.check("alice", 0.0)
    snap = w.snapshot()
    assert snap["enabled"] is True
    assert snap["active_clients"] == 1
    assert snap["overloaded"] is False


def test_distinct_clients_counted_once(monkeypatch):
    monkeypatch.setenv("USAGE_WINDOW", "off")
    w = _fresh_window()
    # mesmo client_id 3x — conta 1
    w.check("alice", 0.0)
    w.check("alice", 0.0)
    w.check("alice", 0.0)
    assert w.snapshot()["active_clients"] == 1


def test_empty_client_id_ignored(monkeypatch):
    monkeypatch.setenv("USAGE_WINDOW", "off")
    w = _fresh_window()
    w.check("", 0.0)
    w.check(None, 0.0)
    assert w.snapshot()["active_clients"] == 0


def test_overload_pauses_heavy_user_when_enabled(monkeypatch):
    """Sanity: a lógica de pausa continua funcionando após o refactor."""
    from fastapi import HTTPException

    monkeypatch.setenv("USAGE_WINDOW", "on")
    monkeypatch.setenv("USAGE_WINDOW_MAX_CLIENTS", "1")
    monkeypatch.setenv("USAGE_WINDOW_HEAVY_FRACTION", "0.5")
    w = _fresh_window()
    # cliente leve passa
    w.check("light", 0.1)
    # cliente pesado vira o 2º → overload → pausa (0.8 >= 0.5)
    with pytest.raises(HTTPException) as exc:
        w.check("heavy", 0.8)
    assert exc.value.status_code == 429


def test_overload_does_not_pause_when_disabled(monkeypatch):
    """Com USAGE_WINDOW=off, mesmo cliente pesado em sobrecarga NÃO é pausado."""
    monkeypatch.setenv("USAGE_WINDOW", "off")
    monkeypatch.setenv("USAGE_WINDOW_MAX_CLIENTS", "1")
    w = _fresh_window()
    w.check("light", 0.1)
    # não levanta — janela off
    w.check("heavy", 0.99)
    assert w.snapshot()["active_clients"] == 2
