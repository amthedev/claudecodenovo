"""
admin_db.py — SQLite admin database para o proxy LLM.
Suporta gestao de chaves, analytics de uso, revenue tracking e reveals seguros.
"""
from __future__ import annotations

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

SESSION_TTL   = int(os.getenv("ADMIN_SESSION_SECONDS", "43200"))
REVEAL_TTL    = int(os.getenv("KEY_REVEAL_TTL", "600"))   # 10 min para copiar a chave
PBKDF2_ITERS  = 260_000


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


# ── Schema ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    with _db() as c:
        c.executescript("""
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
                description TEXT DEFAULT '',
                key_hash TEXT UNIQUE NOT NULL,
                key_preview TEXT NOT NULL,
                daily_limit INTEGER NOT NULL DEFAULT 0,
                monthly_limit INTEGER NOT NULL DEFAULT 0,
                price_per_1k REAL NOT NULL DEFAULT 0.0,
                active INTEGER NOT NULL DEFAULT 1,
                expires_at REAL,
                validity_days INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL,
                last_used_at REAL,
                notes TEXT DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS key_usage (
                key_id TEXT NOT NULL,
                date TEXT NOT NULL,
                request_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (key_id, date),
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS key_reveals (
                token TEXT PRIMARY KEY,
                key_value TEXT NOT NULL,
                key_id TEXT NOT NULL,
                key_name TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'create',
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS key_reveals (
                token TEXT PRIMARY KEY,
                key_value TEXT NOT NULL,
                key_id TEXT NOT NULL,
                key_name TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'create',
                expires_at REAL NOT NULL
            );
        """)
        # Migracoes: adiciona colunas novas em tabelas existentes (seguro se ja existirem)
        new_cols = [
            ("api_keys", "price_per_1k",   "REAL NOT NULL DEFAULT 0.0"),
            ("api_keys", "description",     "TEXT DEFAULT ''"),
            ("api_keys", "monthly_limit",   "INTEGER NOT NULL DEFAULT 0"),
            ("api_keys", "notes",           "TEXT DEFAULT ''"),
        ]
        for table, col, typedef in new_cols:
            try:
                c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typedef}")
            except Exception:
                pass  # coluna ja existe


# ── Hashing ───────────────────────────────────────────────────────────────────

def _hash_pw(pw: str, salt: Optional[str] = None) -> Dict[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), PBKDF2_ITERS).hex()
    return {"salt": salt, "hash": digest}


def _verify_pw(pw: str, h: str, s: str) -> bool:
    try:
        return hmac.compare_digest(_hash_pw(pw, s)["hash"], h)
    except Exception:
        return False


def hash_api_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_sk_key() -> str:
    return "sk-" + secrets.token_urlsafe(32).replace("_", "").replace("-", "")[:42]


# ── Admin auth ────────────────────────────────────────────────────────────────

def admin_exists() -> bool:
    with _db() as c:
        return c.execute("SELECT COUNT(*) FROM admins").fetchone()[0] > 0


def create_admin(username: str, password: str) -> bool:
    if admin_exists():
        return False
    ph = _hash_pw(password)
    with _db() as c:
        c.execute(
            "INSERT INTO admins (id,username,password_hash,password_salt,created_at) VALUES(?,?,?,?,?)",
            (str(uuid.uuid4()), username, ph["hash"], ph["salt"], time.time()),
        )
    return True


def verify_admin(username: str, password: str) -> bool:
    with _db() as c:
        r = c.execute("SELECT password_hash,password_salt FROM admins WHERE username=?", (username,)).fetchone()
    return bool(r and _verify_pw(password, r["password_hash"], r["password_salt"]))


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at<?", (now,))
        c.execute("INSERT INTO sessions(token,username,created_at,expires_at) VALUES(?,?,?,?)",
                  (token, username, now, now + SESSION_TTL))
    return token


def validate_session(token: str) -> Optional[str]:
    if not token:
        return None
    now = time.time()
    with _db() as c:
        r = c.execute("SELECT username,expires_at FROM sessions WHERE token=?", (token,)).fetchone()
        if not r or r["expires_at"] < now:
            if r:
                c.execute("DELETE FROM sessions WHERE token=?", (token,))
            return None
        return r["username"]


def delete_session(token: str) -> None:
    with _db() as c:
        c.execute("DELETE FROM sessions WHERE token=?", (token,))


# ── Key reveals (mostra a chave completa por 10 min) ─────────────────────────

def store_reveal(key_value: str, key_id: str, key_name: str, action: str = "create") -> str:
    """Salva a chave em texto claro temporariamente. Retorna token de reveal."""
    token = secrets.token_urlsafe(24)
    with _db() as c:
        c.execute("DELETE FROM key_reveals WHERE expires_at<?", (time.time(),))
        c.execute(
            "INSERT INTO key_reveals(token,key_value,key_id,key_name,action,expires_at) VALUES(?,?,?,?,?,?)",
            (token, key_value, key_id, key_name, action, time.time() + REVEAL_TTL),
        )
    return token


def get_reveal(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with _db() as c:
        r = c.execute("SELECT * FROM key_reveals WHERE token=? AND expires_at>?",
                      (token, time.time())).fetchone()
        return dict(r) if r else None


def dismiss_reveal(token: str) -> None:
    with _db() as c:
        c.execute("DELETE FROM key_reveals WHERE token=?", (token,))


# ── API Keys ──────────────────────────────────────────────────────────────────

def create_api_key(
    name: str,
    description: str = "",
    daily_limit: int = 0,
    monthly_limit: int = 0,
    validity_days: int = 0,
    price_per_1k: float = 0.0,
    notes: str = "",
) -> tuple[str, str]:
    """Cria chave. Retorna (key_value, reveal_token)."""
    key = generate_sk_key()
    kid = str(uuid.uuid4()).replace("-", "")
    now = time.time()
    expires = (now + validity_days * 86400) if validity_days > 0 else None
    with _db() as c:
        c.execute(
            """INSERT INTO api_keys
               (id,name,description,key_hash,key_preview,daily_limit,monthly_limit,
                price_per_1k,active,expires_at,validity_days,created_at,updated_at,notes)
               VALUES(?,?,?,?,?,?,?,?,1,?,?,?,?,?)""",
            (kid, name, description, hash_api_key(key), key[:14] + "...",
             daily_limit, monthly_limit, price_per_1k, expires, validity_days, now, now, notes),
        )
    reveal_token = store_reveal(key, kid, name, "create")
    return key, reveal_token


def list_api_keys() -> List[Dict[str, Any]]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    month = time.strftime("%Y-%m", time.gmtime())
    with _db() as c:
        rows = c.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            day_u = c.execute("SELECT request_count FROM key_usage WHERE key_id=? AND date=?",
                               (r["id"], today)).fetchone()
            mon_u = c.execute(
                "SELECT COALESCE(SUM(request_count),0) AS cnt FROM key_usage WHERE key_id=? AND date LIKE ?",
                (r["id"], month + "%")).fetchone()
            total_u = c.execute(
                "SELECT COALESCE(SUM(request_count),0) AS cnt FROM key_usage WHERE key_id=?",
                (r["id"],)).fetchone()
            today_count = day_u["request_count"] if day_u else 0
            month_count = mon_u["cnt"] if mon_u else 0
            total_count = total_u["cnt"] if total_u else 0
            revenue = round(total_count / 1000 * r["price_per_1k"], 4)
            result.append({
                "id": r["id"], "name": r["name"], "description": r["description"],
                "key_preview": r["key_preview"], "daily_limit": r["daily_limit"],
                "monthly_limit": r["monthly_limit"], "price_per_1k": r["price_per_1k"],
                "active": bool(r["active"]), "expires_at": r["expires_at"],
                "validity_days": r["validity_days"], "created_at": r["created_at"],
                "last_used_at": r["last_used_at"], "notes": r["notes"] or "",
                "usage_today": today_count, "usage_month": month_count,
                "usage_total": total_count, "revenue_total": revenue,
            })
        return result


def get_api_key(kid: str) -> Optional[Dict[str, Any]]:
    with _db() as c:
        r = c.execute("SELECT * FROM api_keys WHERE id=?", (kid,)).fetchone()
        return dict(r) if r else None


def update_api_key(kid: str, **kwargs) -> bool:
    allowed = {"name", "description", "daily_limit", "monthly_limit",
                "price_per_1k", "active", "notes", "expires_at"}
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return False
    updates["updated_at"] = time.time()
    if "active" in updates:
        updates["active"] = 1 if updates["active"] else 0
    clause = ", ".join(f"{k}=?" for k in updates)
    with _db() as c:
        cur = c.execute(f"UPDATE api_keys SET {clause} WHERE id=?",
                        [*updates.values(), kid])
        return cur.rowcount > 0


def rotate_api_key(kid: str) -> tuple[str, str]:
    """Gera nova chave. Retorna (key_value, reveal_token)."""
    new_key = generate_sk_key()
    info = get_api_key(kid)
    name = info["name"] if info else "key"
    with _db() as c:
        c.execute("UPDATE api_keys SET key_hash=?,key_preview=?,updated_at=? WHERE id=?",
                  (hash_api_key(new_key), new_key[:14] + "...", time.time(), kid))
    reveal_token = store_reveal(new_key, kid, name, "rotate")
    return new_key, reveal_token


def delete_api_key(kid: str) -> bool:
    with _db() as c:
        cur = c.execute("DELETE FROM api_keys WHERE id=?", (kid,))
        return cur.rowcount > 0


def verify_api_key_db(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    kh = hash_api_key(raw)
    today = time.strftime("%Y-%m-%d", time.gmtime())
    now = time.time()
    with _db() as c:
        r = c.execute("SELECT * FROM api_keys WHERE key_hash=?", (kh,)).fetchone()
        if not r:
            return None
        if not r["active"]:
            return {"error": "disabled"}
        if r["expires_at"] and r["expires_at"] <= now:
            return {"error": "expired"}
        dl = r["daily_limit"]
        if dl > 0:
            u = c.execute("SELECT request_count FROM key_usage WHERE key_id=? AND date=?",
                           (r["id"], today)).fetchone()
            if u and u["request_count"] >= dl:
                return {"error": "limit_exceeded"}
        c.execute(
            """INSERT INTO key_usage(key_id,date,request_count) VALUES(?,?,1)
               ON CONFLICT(key_id,date) DO UPDATE SET request_count=request_count+1""",
            (r["id"], today))
        c.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (now, r["id"]))
        return {"type": "managed", "app_id": r["id"], "app_name": r["name"],
                "key_preview": r["key_preview"]}


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    month = time.strftime("%Y-%m", time.gmtime())
    with _db() as c:
        active = c.execute("SELECT COUNT(*) FROM api_keys WHERE active=1").fetchone()[0]
        today_r = c.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage WHERE date=?", (today,)).fetchone()[0]
        month_r = c.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage WHERE date LIKE ?",
            (month + "%",)).fetchone()[0]
        total_r = c.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage").fetchone()[0]
        revenue = c.execute("""
            SELECT COALESCE(SUM(ku.request_count * ak.price_per_1k / 1000.0),0)
            FROM key_usage ku JOIN api_keys ak ON ku.key_id=ak.id
        """).fetchone()[0]
    return {
        "active_keys": active, "today_requests": today_r,
        "month_requests": month_r, "total_requests": total_r,
        "total_revenue": round(revenue, 2), "date": today,
    }


def get_usage_chart(days: int = 14) -> List[Dict[str, Any]]:
    """Retorna uso diario dos ultimos N dias para o grafico."""
    result = []
    for i in range(days - 1, -1, -1):
        ts = time.time() - i * 86400
        d = time.strftime("%Y-%m-%d", time.gmtime(ts))
        with _db() as c:
            cnt = c.execute(
                "SELECT COALESCE(SUM(request_count),0) FROM key_usage WHERE date=?", (d,)
            ).fetchone()[0]
        result.append({"date": d, "label": time.strftime("%d/%m", time.gmtime(ts)), "count": cnt})
    return result


def get_key_usage_history(kid: str, days: int = 14) -> List[Dict[str, Any]]:
    result = []
    for i in range(days - 1, -1, -1):
        ts = time.time() - i * 86400
        d = time.strftime("%Y-%m-%d", time.gmtime(ts))
        with _db() as c:
            r = c.execute("SELECT request_count FROM key_usage WHERE key_id=? AND date=?",
                           (kid, d)).fetchone()
        result.append({"date": d, "label": time.strftime("%d/%m", time.gmtime(ts)),
                       "count": r["request_count"] if r else 0})
    return result


# ── Migracao do JSON antigo ───────────────────────────────────────────────────

def migrate_from_json(json_path: Path) -> int:
    import json as _json
    if not json_path.exists():
        return 0
    try:
        data = _json.loads(json_path.read_text())
    except Exception:
        return 0
    migrated = 0
    admin = data.get("admin")
    if admin and not admin_exists():
        with _db() as c:
            c.execute(
                "INSERT OR IGNORE INTO admins(id,username,password_hash,password_salt,created_at) VALUES(?,?,?,?,?)",
                (str(uuid.uuid4()), admin.get("username", "admin"),
                 admin.get("password", {}).get("hash", ""),
                 admin.get("password", {}).get("salt", ""),
                 admin.get("created_at", time.time())),
            )
    for app in data.get("apps", []):
        kh = app.get("key_hash", "")
        if not kh:
            continue
        try:
            with _db() as c:
                c.execute("""INSERT OR IGNORE INTO api_keys
                   (id,name,key_hash,key_preview,daily_limit,active,expires_at,
                    validity_days,created_at,updated_at,last_used_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                   (app.get("id", str(uuid.uuid4()).replace("-", "")),
                    app.get("name", "migrated"), kh,
                    app.get("key_preview", "sk-..."),
                    app.get("daily_limit", 0),
                    1 if app.get("active", True) else 0,
                    app.get("expires_at"), app.get("validity_days", 0),
                    app.get("created_at", time.time()),
                    app.get("updated_at", time.time()),
                    app.get("last_used_at")))
            migrated += 1
        except Exception:
            pass
    return migrated
