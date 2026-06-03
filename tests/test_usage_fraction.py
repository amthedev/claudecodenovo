import sys
import time
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from proxy_app import admin_db


def _setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    admin_db.init_db()


def _create_key(name: str, *, daily_limit: int = 0, token_limit: int = 0,
                key_type: str = "client") -> str:
    kid = f"id-{name}"
    with admin_db._db() as c:
        c.execute(
            """INSERT INTO api_keys(id, name, key_hash, key_preview, daily_limit,
                                    monthly_limit, token_limit, key_type, active,
                                    created_at)
               VALUES(?, ?, ?, 'p', ?, 0, ?, ?, 1, ?)""",
            (kid, name, f"h-{name}", daily_limit, token_limit, key_type, time.time()),
        )
    return kid


def _insert_usage(key_id: str, tokens: int, date: str = None) -> None:
    date = date or admin_db._today_str()
    with admin_db._db() as c:
        c.execute(
            """INSERT INTO key_usage(key_id, date, request_count, input_tokens,
                                     output_tokens, total_tokens)
               VALUES(?, ?, 1, 0, 0, ?)
               ON CONFLICT(key_id, date) DO UPDATE SET
                 total_tokens = total_tokens + excluded.total_tokens""",
            (key_id, date, tokens),
        )


def test_unknown_key_returns_zero(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    assert admin_db.daily_usage_fraction("nao-existe") == 0.0


def test_key_with_no_limits_returns_zero(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _create_key("sem-limite", daily_limit=0, token_limit=0)
    _insert_usage("id-sem-limite", 500)
    assert admin_db.daily_usage_fraction("sem-limite") == 0.0


def test_daily_limit_used_when_set(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    _create_key("diario", daily_limit=100, token_limit=0)
    _insert_usage("id-diario", 50)
    assert admin_db.daily_usage_fraction("diario") == 0.5


def test_token_limit_used_when_daily_limit_is_zero(tmp_path, monkeypatch):
    """Regressão: antes, chave só-com-token_limit retornava 0.0 e a janela
    nunca pausava ninguém. Agora deve usar token_limit como fallback."""
    _setup(tmp_path, monkeypatch)
    _create_key("pacote", daily_limit=0, token_limit=100)
    _insert_usage("id-pacote", 50)
    assert admin_db.daily_usage_fraction("pacote") == 0.5


def test_daily_takes_precedence_over_token_limit(tmp_path, monkeypatch):
    """Quando os dois estão setados, daily_limit ganha (mais restritivo
    intencional)."""
    _setup(tmp_path, monkeypatch)
    _create_key("ambos", daily_limit=100, token_limit=1000)
    _insert_usage("id-ambos", 50)
    assert admin_db.daily_usage_fraction("ambos") == 0.5
