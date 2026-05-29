"""
admin_db.py - Sistema de admin baseado em SQLite para o proxy LLM.
Substitui o admin_data.json por um banco relacional com suporte a WAL.
"""

import hashlib
import hmac
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


def _db_path() -> Path:
    return Path.cwd() / os.getenv("ADMIN_DB_FILE", "admin.db")


@contextmanager
def _db():
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Cria as tabelas se ainda nao existirem."""
    with _db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS admins (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_preview TEXT NOT NULL,
                daily_limit INTEGER NOT NULL DEFAULT 0,
                active INTEGER NOT NULL DEFAULT 1,
                expires_at REAL,
                validity_days INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL,
                last_used_at REAL
            );

            CREATE TABLE IF NOT EXISTS key_usage (
                key_id TEXT NOT NULL,
                date TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key_id, date),
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
        """)


# ── Hashing ───────────────────────────────────────────────────────────────────

_PBKDF2_ITERS = 260_000
SESSION_TTL = int(os.getenv("ADMIN_SESSION_SECONDS", "43200"))


def _hash_password(password: str, salt: Optional[str] = None) -> Dict[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ITERS
    ).hex()
    return {"salt": salt, "hash": digest}


def _verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    candidate = _hash_password(password, stored_salt)
    return hmac.compare_digest(candidate["hash"], stored_hash)


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_sk_key() -> str:
    return "sk-" + secrets.token_urlsafe(32).replace("_", "").replace("-", "")[:42]


# ── Admin auth ────────────────────────────────────────────────────────────────

def admin_exists() -> bool:
    with _db() as conn:
        return conn.execute("SELECT COUNT(*) FROM admins").fetchone()[0] > 0


def create_admin(username: str, password: str) -> bool:
    if admin_exists():
        return False
    ph = _hash_password(password)
    with _db() as conn:
        conn.execute(
            "INSERT INTO admins (id,username,password_hash,password_salt,created_at) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), username, ph["hash"], ph["salt"], time.time()),
        )
    return True


def verify_admin(username: str, password: str) -> bool:
    with _db() as conn:
        row = conn.execute(
            "SELECT password_hash, password_salt FROM admins WHERE username=?", (username,)
        ).fetchone()
    if not row:
        return False
    return _verify_password(password, row["password_hash"], row["password_salt"])


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        conn.execute(
            "INSERT INTO sessions (token,username,created_at,expires_at) VALUES (?,?,?,?)",
            (token, username, now, now + SESSION_TTL),
        )
    return token


def validate_session(token: str) -> Optional[str]:
    if not token:
        return None
    now = time.time()
    with _db() as conn:
        row = conn.execute(
            "SELECT username, expires_at FROM sessions WHERE token=?", (token,)
        ).fetchone()
        if not row or row["expires_at"] < now:
            if row:
                conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            return None
        return row["username"]


def delete_session(token: str) -> None:
    with _db() as conn:
        conn.execute("DELETE FROM sessions WHERE token=?", (token,))


# ── API Keys ──────────────────────────────────────────────────────────────────

def create_api_key(name: str, daily_limit: int = 0, validity_days: int = 0) -> str:
    """Cria nova chave sk- e retorna o valor em texto claro (salvo apenas o hash)."""
    key = generate_sk_key()
    key_id = str(uuid.uuid4()).replace("-", "")
    now = time.time()
    expires_at = (now + validity_days * 86400) if validity_days > 0 else None
    with _db() as conn:
        conn.execute(
            """INSERT INTO api_keys
               (id,name,key_hash,key_preview,daily_limit,active,expires_at,validity_days,created_at,updated_at)
               VALUES (?,?,?,?,?,1,?,?,?,?)""",
            (key_id, name, hash_api_key(key), key[:14] + "...",
             daily_limit, expires_at, validity_days, now, now),
        )
    return key


def list_api_keys() -> List[Dict[str, Any]]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with _db() as conn:
        rows = conn.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            usage = conn.execute(
                "SELECT request_count FROM key_usage WHERE key_id=? AND date=?",
                (r["id"], today),
            ).fetchone()
            result.append({
                "id": r["id"],
                "name": r["name"],
                "key_preview": r["key_preview"],
                "daily_limit": r["daily_limit"],
                "active": bool(r["active"]),
                "expires_at": r["expires_at"],
                "validity_days": r["validity_days"],
                "created_at": r["created_at"],
                "last_used_at": r["last_used_at"],
                "usage_today": usage["request_count"] if usage else 0,
            })
        return result


def get_api_key(key_id: str) -> Optional[Dict[str, Any]]:
    with _db() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        return dict(row) if row else None


def update_api_key(key_id: str, **kwargs) -> bool:
    allowed = {"name", "daily_limit", "active"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    updates["updated_at"] = time.time()
    if "active" in updates:
        updates["active"] = 1 if updates["active"] else 0
    set_clause = ", ".join(f"{k}=?" for k in updates)
    vals = list(updates.values()) + [key_id]
    with _db() as conn:
        cur = conn.execute(f"UPDATE api_keys SET {set_clause} WHERE id=?", vals)
        return cur.rowcount > 0


def rotate_api_key(key_id: str) -> Optional[str]:
    new_key = generate_sk_key()
    with _db() as conn:
        cur = conn.execute(
            "UPDATE api_keys SET key_hash=?, key_preview=?, updated_at=? WHERE id=?",
            (hash_api_key(new_key), new_key[:14] + "...", time.time(), key_id),
        )
        return new_key if cur.rowcount > 0 else None


def delete_api_key(key_id: str) -> bool:
    with _db() as conn:
        cur = conn.execute("DELETE FROM api_keys WHERE id=?", (key_id,))
        return cur.rowcount > 0


def verify_api_key_db(raw_key: str) -> Optional[Dict[str, Any]]:
    """Verifica a chave e incrementa o contador de uso. Retorna info ou None."""
    if not raw_key:
        return None
    key_hash = hash_api_key(raw_key)
    today = time.strftime("%Y-%m-%d", time.gmtime())
    now = time.time()

    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash=?", (key_hash,)
        ).fetchone()
        if not row:
            return None
        if not row["active"]:
            return {"error": "disabled"}
        if row["expires_at"] and row["expires_at"] <= now:
            return {"error": "expired"}

        daily_limit = row["daily_limit"]
        if daily_limit > 0:
            usage = conn.execute(
                "SELECT request_count FROM key_usage WHERE key_id=? AND date=?",
                (row["id"], today),
            ).fetchone()
            if usage and usage["request_count"] >= daily_limit:
                return {"error": "limit_exceeded"}

        conn.execute(
            """INSERT INTO key_usage (key_id,date,request_count) VALUES (?,?,1)
               ON CONFLICT(key_id,date) DO UPDATE SET request_count=request_count+1""",
            (row["id"], today),
        )
        conn.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (now, row["id"]))

        return {
            "type": "managed",
            "app_id": row["id"],
            "app_name": row["name"],
            "key_preview": row["key_preview"],
        }


def get_stats() -> Dict[str, Any]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    with _db() as conn:
        active_keys = conn.execute(
            "SELECT COUNT(*) FROM api_keys WHERE active=1"
        ).fetchone()[0]
        today_req = conn.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage WHERE date=?", (today,)
        ).fetchone()[0]
        total_req = conn.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage"
        ).fetchone()[0]
    return {
        "active_keys": active_keys,
        "today_requests": today_req,
        "total_requests": total_req,
        "date": today,
    }


# ── Migracao do JSON antigo ───────────────────────────────────────────────────

def migrate_from_json(json_path: Path) -> int:
    """Migra chaves do admin_data.json para o SQLite. Retorna quantas foram migradas."""
    import json
    if not json_path.exists():
        return 0
    try:
        data = json.loads(json_path.read_text())
    except Exception:
        return 0

    migrated = 0

    # Migra admin
    admin = data.get("admin")
    if admin and not admin_exists():
        with _db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO admins (id,username,password_hash,password_salt,created_at) VALUES (?,?,?,?,?)",
                (str(uuid.uuid4()), admin.get("username","admin"),
                 admin.get("password",{}).get("hash",""),
                 admin.get("password",{}).get("salt",""),
                 admin.get("created_at", time.time())),
            )

    # Migra chaves
    for app in data.get("apps", []):
        key_hash = app.get("key_hash","")
        if not key_hash:
            continue
        try:
            with _db() as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO api_keys
                       (id,name,key_hash,key_preview,daily_limit,active,expires_at,validity_days,created_at,updated_at,last_used_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                    (app.get("id", str(uuid.uuid4()).replace("-","")),
                     app.get("name","migrated"),
                     key_hash,
                     app.get("key_preview","sk-..."),
                     app.get("daily_limit",0),
                     1 if app.get("active",True) else 0,
                     app.get("expires_at"),
                     app.get("validity_days",0),
                     app.get("created_at", time.time()),
                     app.get("updated_at", time.time()),
                     app.get("last_used_at")),
                )
                migrated += 1
        except Exception:
            pass

    return migrated
