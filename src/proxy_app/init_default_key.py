#!/usr/bin/env python3
"""
init_default_key.py
Auto-cria uma chave universal sk-xxxx no admin_data.json ao iniciar.
Se a chave já existir, apenas confirma e segue em frente.
"""

import hashlib
import json
import os
import secrets
import sys
import time
import uuid
from pathlib import Path

# Mesma lógica de _root_dir do main.py
if getattr(sys, "frozen", False):
    _root_dir = Path(sys.executable).parent
else:
    _root_dir = Path.cwd()

ADMIN_DATA_FILE = _root_dir / os.getenv("ADMIN_DATA_FILE", "admin_data.json")
UNIVERSAL_KEY_NAME = "universal"
DAILY_LIMIT = 500_000_000  # 500 milhões de requisições/dia (efetivamente ilimitado)


# Use the canonical helpers from auth_helpers/admin_db. Previously this file
# had its own copies which drifted from the originals — same bug shape, two
# places to fix. Re-export the shared ones to keep call sites unchanged.
try:
    from proxy_app.auth_helpers import hash_api_key as _hash_api_key
    from proxy_app.auth_helpers import generate_proxy_key as _generate_sk_key
except ImportError:
    # Fallback: this script is sometimes run before the package is fully
    # importable (initial bootstrap). Keep local definitions as a safety net.
    def _hash_api_key(api_key: str) -> str:
        return hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    def _generate_sk_key() -> str:
        return "sk-" + secrets.token_urlsafe(32).replace("_", "").replace("-", "")[:42]


def _load_admin_data() -> dict:
    if not ADMIN_DATA_FILE.exists():
        return {"admin": None, "sessions": {}, "apps": []}
    try:
        with ADMIN_DATA_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("admin", None)
        data.setdefault("sessions", {})
        data.setdefault("apps", [])
        return data
    except Exception:
        return {"admin": None, "sessions": {}, "apps": []}


def _save_admin_data(data: dict) -> None:
    ADMIN_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ADMIN_DATA_FILE.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp_path.replace(ADMIN_DATA_FILE)


def _ensure_active_in_sqlite(name: str) -> bool:
    """Re-ativa a chave no SQLite se ela existir mas estiver desativada. Retorna True se encontrou."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from proxy_app import admin_db
        admin_db.init_db()
        with admin_db._db() as c:
            r = c.execute("SELECT id, active FROM api_keys WHERE name=?", (name,)).fetchone()
            if r:
                if not r["active"]:
                    c.execute("UPDATE api_keys SET active=1, updated_at=? WHERE id=?",
                              (time.time(), r["id"]))
                    print(f"[init_key] Chave '{name}' reativada no SQLite.")
                else:
                    print(f"[init_key] Chave '{name}' já ativa no SQLite.")
                return True
        return False
    except Exception as e:
        print(f"[init_key] Aviso: não foi possível verificar SQLite: {e}")
        return False


def ensure_universal_key() -> None:
    # Primeiro tenta garantir que a chave existe e está ativa no SQLite
    if _ensure_active_in_sqlite(UNIVERSAL_KEY_NAME):
        return

    data = _load_admin_data()

    # Verifica se já existe uma chave "universal" no JSON
    for app in data.get("apps", []):
        if app.get("name") == UNIVERSAL_KEY_NAME:
            print(f"[init_key] Chave universal já existe no JSON: {app['key_preview']}")
            return

    # Gera nova chave sk-xxxx
    key = _generate_sk_key()
    key_id = str(uuid.uuid4())

    app_entry = {
        "active": True,
        "created_at": time.time(),
        "daily_limit": DAILY_LIMIT,
        "expires_at": None,
        "id": key_id,
        "key_hash": _hash_api_key(key),
        "key_preview": key[:14] + "...",
        "last_used_at": None,
        "name": UNIVERSAL_KEY_NAME,
        "usage": {},
    }

    data["apps"].append(app_entry)
    _save_admin_data(data)

    print("=" * 60)
    print("[init_key] CHAVE UNIVERSAL CRIADA:")
    print(f"[init_key] {key}")
    print(f"[init_key] Limite diario: {DAILY_LIMIT:,} requisicoes")
    print("[init_key] Use essa chave como: Authorization: Bearer <chave>")
    print("=" * 60)


if __name__ == "__main__":
    ensure_universal_key()
