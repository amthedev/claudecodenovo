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

# Janela do "dia" para limites diários. Default America/Sao_Paulo (UTC-3, sem
# horário de verão) — antes era UTC e o reset caía às 21h locais. Configurável
# via env DAILY_RESET_OFFSET_HOURS (em horas, positivo = oeste de UTC).
_DAILY_OFFSET_SEC = int(float(os.getenv("DAILY_RESET_OFFSET_HOURS", "3")) * 3600)


def _local_struct(ts: Optional[float] = None) -> time.struct_time:
    """time.struct_time deslocado pro fuso local (default Brasília)."""
    return time.gmtime((ts if ts is not None else time.time()) - _DAILY_OFFSET_SEC)


def _today_str(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%d", _local_struct(ts))


def _month_str(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m", _local_struct(ts))


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
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
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
            CREATE TABLE IF NOT EXISTS reseller_sessions (
                token TEXT PRIMARY KEY,
                key_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS web_accounts (
                id TEXT PRIMARY KEY,
                key_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS web_sessions (
                token_hash TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                key_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (account_id) REFERENCES web_accounts(id) ON DELETE CASCADE,
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS client_sessions (
                token_hash TEXT PRIMARY KEY,
                key_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                FOREIGN KEY (key_id) REFERENCES api_keys(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS drive_automations (
                id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL,
                name TEXT NOT NULL,
                webhook_url TEXT NOT NULL,
                action TEXT NOT NULL,
                source_folder TEXT DEFAULT '',
                destination_folder TEXT DEFAULT '',
                file_pattern TEXT DEFAULT '',
                created_at REAL NOT NULL,
                FOREIGN KEY (account_id) REFERENCES web_accounts(id) ON DELETE CASCADE
            );
        """)
        # Migracoes: adiciona colunas novas em tabelas existentes (seguro se ja existirem)
        new_cols = [
            ("api_keys", "price_per_1k",   "REAL NOT NULL DEFAULT 0.0"),
            ("api_keys", "description",     "TEXT DEFAULT ''"),
            ("api_keys", "monthly_limit",   "INTEGER NOT NULL DEFAULT 0"),
            ("api_keys", "notes",           "TEXT DEFAULT ''"),
            ("api_keys", "key_type",        "TEXT NOT NULL DEFAULT 'client'"),
            ("api_keys", "parent_key_id",   "TEXT"),
            ("api_keys", "token_limit",     "INTEGER NOT NULL DEFAULT 0"),
            ("key_usage", "input_tokens",   "INTEGER NOT NULL DEFAULT 0"),
            ("key_usage", "output_tokens",  "INTEGER NOT NULL DEFAULT 0"),
            ("key_usage", "total_tokens",   "INTEGER NOT NULL DEFAULT 0"),
            # Reseller accounts: web_accounts gains a role + approval status so the
            # owner can run a signup→approve→suspend flow with email/password login.
            ("web_accounts", "role",        "TEXT NOT NULL DEFAULT 'client'"),
            ("web_accounts", "status",      "TEXT NOT NULL DEFAULT 'active'"),
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


def generate_proxy_key() -> str:
    """Gera uma chave root no formato proxy_xxxx."""
    return "proxy_" + secrets.token_urlsafe(32).replace("_", "").replace("-", "")[:42]


# ── App settings (override de chave root) ───────────────────────────────────────

def _ensure_settings_table(c) -> None:
    c.execute(
        "CREATE TABLE IF NOT EXISTS app_settings (key TEXT PRIMARY KEY, value TEXT)"
    )


def set_root_key_override(new_key: str) -> str:
    """Armazena o hash da nova chave root (precede a env var). Retorna reveal token."""
    h = hash_api_key(new_key)
    preview = new_key[:18] + "..."
    with _db() as c:
        _ensure_settings_table(c)
        c.execute(
            "INSERT OR REPLACE INTO app_settings(key,value) VALUES('root_key_hash',?)",
            (h,),
        )
        c.execute(
            "INSERT OR REPLACE INTO app_settings(key,value) VALUES('root_key_preview',?)",
            (preview,),
        )
    return store_reveal(new_key, "root", "Chave Root (proxy)", "rotate")


def get_root_key_hash() -> Optional[str]:
    with _db() as c:
        try:
            _ensure_settings_table(c)
            r = c.execute(
                "SELECT value FROM app_settings WHERE key='root_key_hash'"
            ).fetchone()
            return r["value"] if r else None
        except Exception:
            return None


def get_root_key_preview() -> Optional[str]:
    with _db() as c:
        try:
            _ensure_settings_table(c)
            r = c.execute(
                "SELECT value FROM app_settings WHERE key='root_key_preview'"
            ).fetchone()
            return r["value"] if r else None
        except Exception:
            return None


# Thinking mode persistido (sobrevive a multi-worker; o global em memória do main
# divergia entre workers do gunicorn — flag "/think on" só funcionava em 1/N).
_VALID_THINKING_MODES = {"on", "off", "auto"}


def get_thinking_mode() -> Optional[str]:
    """Returns the user-set thinking mode, or None if the user never picked one.

    Returning None lets the translator distinguish 'user explicitly chose off'
    from 'user never said anything, follow env default'. UI surfaces that want
    a string for display can do `get_thinking_mode() or "auto"`."""
    with _db() as c:
        try:
            _ensure_settings_table(c)
            r = c.execute(
                "SELECT value FROM app_settings WHERE key='thinking_mode'"
            ).fetchone()
            v = r["value"] if r else None
            return v if v in _VALID_THINKING_MODES else None
        except Exception:
            return None


def set_thinking_mode(mode: str) -> str:
    if mode not in _VALID_THINKING_MODES:
        mode = "off"
    with _db() as c:
        _ensure_settings_table(c)
        c.execute(
            "INSERT OR REPLACE INTO app_settings(key,value) VALUES('thinking_mode',?)",
            (mode,),
        )
    return mode


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
        # Defense-in-depth: the session must point at a real admin row. The
        # previous version trusted any (token, username) tuple — if a session
        # was minted for an arbitrary string (the setup_post escalation),
        # _need accepted it. Cross-check against the admins table here.
        admin_row = c.execute(
            "SELECT 1 FROM admins WHERE username=?", (r["username"],)
        ).fetchone()
        if not admin_row:
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
    key_type: str = "client",
    parent_key_id: Optional[str] = None,
    token_limit: int = 0,
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
                price_per_1k,active,expires_at,validity_days,created_at,updated_at,notes,
                key_type,parent_key_id,token_limit)
               VALUES(?,?,?,?,?,?,?,?,1,?,?,?,?,?,?,?,?)""",
            (kid, name, description, hash_api_key(key), key[:14] + "...",
             daily_limit, monthly_limit, price_per_1k, expires, validity_days, now, now, notes,
             key_type, parent_key_id, token_limit),
        )
    reveal_token = store_reveal(key, kid, name, "create")
    return key, reveal_token


def list_api_keys() -> List[Dict[str, Any]]:
    today = _today_str()
    month = _month_str()
    with _db() as c:
        rows = c.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
        result = []
        for r in rows:
            day_u = c.execute("SELECT request_count,total_tokens FROM key_usage WHERE key_id=? AND date=?",
                               (r["id"], today)).fetchone()
            mon_u = c.execute(
                "SELECT COALESCE(SUM(request_count),0) AS cnt, COALESCE(SUM(total_tokens),0) AS tokens FROM key_usage WHERE key_id=? AND date LIKE ?",
                (r["id"], month + "%")).fetchone()
            total_u = c.execute(
                "SELECT COALESCE(SUM(request_count),0) AS cnt, COALESCE(SUM(total_tokens),0) AS tokens FROM key_usage WHERE key_id=?",
                (r["id"],)).fetchone()
            today_count = day_u["request_count"] if day_u else 0
            today_tokens = day_u["total_tokens"] if day_u else 0
            month_count = mon_u["cnt"] if mon_u else 0
            month_tokens = mon_u["tokens"] if mon_u else 0
            total_count = total_u["cnt"] if total_u else 0
            total_tokens = total_u["tokens"] if total_u else 0
            child_usage = c.execute(
                """SELECT COALESCE(SUM(ku.total_tokens),0) AS tokens
                   FROM key_usage ku JOIN api_keys child ON child.id=ku.key_id
                   WHERE child.parent_key_id=?""", (r["id"],)
            ).fetchone()["tokens"]
            allocated = c.execute(
                "SELECT COALESCE(SUM(token_limit),0) AS tokens FROM api_keys WHERE parent_key_id=?",
                (r["id"],)
            ).fetchone()["tokens"]
            effective_tokens = total_tokens + child_usage
            token_limit = r["token_limit"] or 0
            tokens_remaining = _remaining_tokens_for_row(c, r)
            revenue = round(total_tokens / 1000 * r["price_per_1k"], 4)
            result.append({
                "id": r["id"], "name": r["name"], "description": r["description"],
                "key_preview": r["key_preview"], "daily_limit": r["daily_limit"],
                "monthly_limit": r["monthly_limit"], "price_per_1k": r["price_per_1k"],
                "active": bool(r["active"]), "expires_at": r["expires_at"],
                "validity_days": r["validity_days"], "created_at": r["created_at"],
                "last_used_at": r["last_used_at"], "notes": r["notes"] or "",
                "usage_today": today_count, "usage_month": month_count,
                "usage_total": total_count, "tokens_today": today_tokens,
                "tokens_month": month_tokens, "tokens_total": total_tokens,
                "effective_tokens_total": effective_tokens,
                "token_limit": token_limit,
                "tokens_remaining": max(0, tokens_remaining) if tokens_remaining is not None else None,
                "access_error": _row_access_error(c, r),
                "allocated_tokens": allocated,
                "key_type": r["key_type"] or "client",
                "parent_key_id": r["parent_key_id"],
                "revenue_total": revenue,
            })
        return result


def has_api_keys() -> bool:
    with _db() as c:
        return c.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0] > 0


def get_api_key(kid: str) -> Optional[Dict[str, Any]]:
    with _db() as c:
        r = c.execute("SELECT * FROM api_keys WHERE id=?", (kid,)).fetchone()
        return dict(r) if r else None


def update_api_key(kid: str, **kwargs) -> bool:
    allowed = {"name", "description", "daily_limit", "monthly_limit",
                "price_per_1k", "active", "notes", "expires_at", "token_limit"}
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


def recharge_api_key(kid: str, add_tokens: int = 0, add_daily_tokens: int = 0,
                     add_monthly_tokens: int = 0, add_validity_days: int = 0,
                     owner_reseller_id: Optional[str] = None) -> Dict[str, Any]:
    """Adiciona franquia a uma chave sem apagar o historico de consumo."""
    add_tokens = max(0, int(add_tokens or 0))
    add_daily_tokens = max(0, int(add_daily_tokens or 0))
    add_monthly_tokens = max(0, int(add_monthly_tokens or 0))
    add_validity_days = max(0, int(add_validity_days or 0))
    with _db() as c:
        row = c.execute("SELECT * FROM api_keys WHERE id=?", (kid,)).fetchone()
        if not row:
            raise ValueError("Chave não encontrada.")
        if owner_reseller_id and row["parent_key_id"] != owner_reseller_id:
            raise ValueError("Esta chave não pertence ao revendedor.")
        if row["parent_key_id"] and add_tokens:
            parent = c.execute(
                "SELECT * FROM api_keys WHERE id=?", (row["parent_key_id"],)
            ).fetchone()
            if not parent or _row_access_error(c, parent):
                raise ValueError("Chave mestre inválida ou inativa.")
            pool = int(parent["token_limit"] or 0)
            allocated = int(c.execute(
                "SELECT COALESCE(SUM(token_limit),0) FROM api_keys WHERE parent_key_id=?",
                (parent["id"],),
            ).fetchone()[0])
            if pool > 0 and allocated + add_tokens > pool:
                raise ValueError("A recarga ultrapassa o saldo distribuível da chave mestre.")
        current_expiry = float(row["expires_at"] or 0)
        expires_at = current_expiry
        if add_validity_days:
            expires_at = max(time.time(), current_expiry) + add_validity_days * 86400
        c.execute(
            """UPDATE api_keys SET token_limit=token_limit+?,
               daily_limit=daily_limit+?, monthly_limit=monthly_limit+?,
               expires_at=?, updated_at=? WHERE id=?""",
            (add_tokens, add_daily_tokens, add_monthly_tokens,
             expires_at or None, time.time(), kid),
        )
    return get_api_key(kid) or {}


def rotate_api_key(kid: str) -> tuple[str, str]:
    """Gera nova chave. Retorna (key_value, reveal_token)."""
    new_key = generate_sk_key()
    info = get_api_key(kid)
    if not info:
        raise ValueError("Chave não encontrada.")
    name = info["name"]
    with _db() as c:
        c.execute("UPDATE api_keys SET key_hash=?,key_preview=?,updated_at=? WHERE id=?",
                  (hash_api_key(new_key), new_key[:14] + "...", time.time(), kid))
    reveal_token = store_reveal(new_key, kid, name, "rotate")
    return new_key, reveal_token


def delete_api_key(kid: str) -> bool:
    with _db() as c:
        c.execute("DELETE FROM key_reveals WHERE key_id=? OR key_id IN (SELECT id FROM api_keys WHERE parent_key_id=?)",
                  (kid, kid))
        c.execute("DELETE FROM api_keys WHERE parent_key_id=?", (kid,))
        cur = c.execute("DELETE FROM api_keys WHERE id=?", (kid,))
        return cur.rowcount > 0


def verify_api_key_db(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    kh = hash_api_key(raw)
    today = _today_str()
    now = time.time()
    with _db() as c:
        r = c.execute("SELECT * FROM api_keys WHERE key_hash=?", (kh,)).fetchone()
        if not r:
            c.execute("DELETE FROM web_sessions WHERE expires_at<=?", (now,))
            r = c.execute(
                """SELECT ak.* FROM web_sessions ws
                   JOIN api_keys ak ON ak.id=ws.key_id
                   WHERE ws.token_hash=? AND ws.expires_at>?""",
                (kh, now),
            ).fetchone()
        if not r:
            return None
        error = _row_access_error(c, r, now=now, today=today)
        if error:
            return {"error": error}
        return {"type": "managed", "app_id": r["id"], "app_name": r["name"],
                "key_preview": r["key_preview"], "key_type": r["key_type"] or "client"}


# ── Contas web ────────────────────────────────────────────────────────────────

def _web_account_payload(account_id: str) -> Optional[Dict[str, Any]]:
    with _db() as c:
        account = c.execute(
            "SELECT id,key_id,name,email,created_at FROM web_accounts WHERE id=?",
            (account_id,),
        ).fetchone()
    if not account:
        return None
    key = next((item for item in list_api_keys() if item["id"] == account["key_id"]), None)
    if not key:
        return None
    return {
        "id": account["id"],
        "name": account["name"],
        "email": account["email"],
        "created_at": account["created_at"],
        "plan": key["name"],
        "active": key["active"],
        "access_status": (verify_api_key_db_by_id(account["key_id"]) or {}).get("error") or "active",
        "expires_at": key["expires_at"],
        "tokens_today": key["tokens_today"],
        "tokens_total": key["tokens_total"],
        "token_limit": key["token_limit"],
        "tokens_remaining": key["tokens_remaining"],
        "daily_limit": key["daily_limit"],
        "monthly_limit": key["monthly_limit"],
    }


def _create_web_session(account_id: str, key_id: str) -> str:
    token = "web-" + secrets.token_urlsafe(36)
    now = time.time()
    with _db() as c:
        c.execute("DELETE FROM web_sessions WHERE expires_at<=?", (now,))
        c.execute(
            """INSERT INTO web_sessions(token_hash,account_id,key_id,created_at,expires_at)
               VALUES(?,?,?,?,?)""",
            (hash_api_key(token), account_id, key_id, now, now + SESSION_TTL),
        )
    return token


def create_client_session(raw_key: str) -> Dict[str, Any]:
    """For the /cliente panel: validate a pasted client key and return an opaque
    session token (so the raw key never lives in a browser cookie). Returns
    {"token", "key_id"} on success or {"error": msg} on failure. Uses its own
    client_sessions table (no web_account needed behind a pasted-key session)."""
    verified = verify_api_key_db((raw_key or "").strip())
    if not verified:
        return {"error": "Chave inválida ou não encontrada."}
    if verified.get("error"):
        return {"error": verified["error"]}
    token = "cli-" + secrets.token_urlsafe(36)
    now = time.time()
    with _db() as c:
        c.execute("DELETE FROM client_sessions WHERE expires_at<=?", (now,))
        c.execute(
            """INSERT INTO client_sessions(token_hash,key_id,created_at,expires_at)
               VALUES(?,?,?,?)""",
            (hash_api_key(token), verified["app_id"], now, now + SESSION_TTL),
        )
    return {"token": token, "key_id": verified["app_id"]}


def client_session_key_id(token: str) -> Optional[str]:
    """Resolve a /cliente session cookie back to its key_id (or None if expired)."""
    if not token:
        return None
    now = time.time()
    with _db() as c:
        c.execute("DELETE FROM client_sessions WHERE expires_at<=?", (now,))
        row = c.execute(
            "SELECT key_id FROM client_sessions WHERE token_hash=? AND expires_at>?",
            (hash_api_key(token), now),
        ).fetchone()
    return row["key_id"] if row else None


def delete_client_session(token: str) -> None:
    """Drop a /cliente session (logout)."""
    if not token:
        return
    with _db() as c:
        c.execute("DELETE FROM client_sessions WHERE token_hash=?", (hash_api_key(token),))


def create_web_account(name: str, email: str, password: str, raw_key: str) -> Dict[str, Any]:
    name = name.strip()
    email = email.strip().lower()
    if not name:
        raise ValueError("Informe seu nome.")
    if "@" not in email:
        raise ValueError("Informe um e-mail válido.")
    if len(password) < 8:
        raise ValueError("A senha precisa ter pelo menos 8 caracteres.")
    verified = verify_api_key_db(raw_key.strip())
    if not verified or verified.get("error") or verified.get("key_type") != "client":
        raise ValueError("Token inválido, expirado ou sem acesso de cliente.")
    account_id = str(uuid.uuid4()).replace("-", "")
    ph = _hash_pw(password)
    try:
        with _db() as c:
            c.execute(
                """INSERT INTO web_accounts
                   (id,key_id,name,email,password_hash,password_salt,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (account_id, verified["app_id"], name, email, ph["hash"], ph["salt"], time.time()),
            )
    except sqlite3.IntegrityError as exc:
        message = str(exc).lower()
        if "email" in message:
            raise ValueError("Este e-mail já possui uma conta.") from exc
        raise ValueError("Este token já foi vinculado a uma conta.") from exc
    return {"account": _web_account_payload(account_id), "access_token": _create_web_session(account_id, verified["app_id"])}


def login_web_account(email: str, password: str) -> Dict[str, Any]:
    email = email.strip().lower()
    with _db() as c:
        account = c.execute("SELECT * FROM web_accounts WHERE email=?", (email,)).fetchone()
    if not account or not _verify_pw(password, account["password_hash"], account["password_salt"]):
        raise ValueError("E-mail ou senha incorretos.")
    return {"account": _web_account_payload(account["id"]), "access_token": _create_web_session(account["id"], account["key_id"])}


def verify_api_key_db_by_id(key_id: str) -> Optional[Dict[str, Any]]:
    with _db() as c:
        key = c.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        if not key:
            return None
        today = _today_str()
        error = _row_access_error(c, key, now=time.time(), today=today)
        if error:
            return {"error": error}
        return {"type": "managed", "app_id": key["id"], "app_name": key["name"],
                "key_preview": key["key_preview"], "key_type": key["key_type"] or "client"}


def get_web_account_by_session(raw_token: str) -> Optional[Dict[str, Any]]:
    if not raw_token:
        return None
    now = time.time()
    with _db() as c:
        c.execute("DELETE FROM web_sessions WHERE expires_at<=?", (now,))
        session = c.execute(
            "SELECT account_id FROM web_sessions WHERE token_hash=? AND expires_at>?",
            (hash_api_key(raw_token), now),
        ).fetchone()
    return _web_account_payload(session["account_id"]) if session else None


def get_web_account_id_by_session(raw_token: str) -> Optional[str]:
    if not raw_token:
        return None
    with _db() as c:
        session = c.execute(
            "SELECT account_id FROM web_sessions WHERE token_hash=? AND expires_at>?",
            (hash_api_key(raw_token), time.time()),
        ).fetchone()
    return session["account_id"] if session else None


def delete_web_session(raw_token: str) -> None:
    if not raw_token:
        return
    with _db() as c:
        c.execute("DELETE FROM web_sessions WHERE token_hash=?", (hash_api_key(raw_token),))


def create_drive_automation(account_id: str, name: str, webhook_url: str, action: str,
                            source_folder: str = "", destination_folder: str = "",
                            file_pattern: str = "") -> Dict[str, Any]:
    automation_id = str(uuid.uuid4()).replace("-", "")
    with _db() as c:
        c.execute(
            """INSERT INTO drive_automations
               (id,account_id,name,webhook_url,action,source_folder,destination_folder,file_pattern,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (automation_id, account_id, name.strip(), webhook_url.strip(), action.strip(),
             source_folder.strip(), destination_folder.strip(), file_pattern.strip(), time.time()),
        )
    return get_drive_automation(account_id, automation_id) or {}


def get_drive_automation(account_id: str, automation_id: str) -> Optional[Dict[str, Any]]:
    with _db() as c:
        row = c.execute(
            "SELECT * FROM drive_automations WHERE id=? AND account_id=?",
            (automation_id, account_id),
        ).fetchone()
    return dict(row) if row else None


def list_drive_automations(account_id: str) -> List[Dict[str, Any]]:
    with _db() as c:
        rows = c.execute(
            "SELECT * FROM drive_automations WHERE account_id=? ORDER BY created_at DESC",
            (account_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def delete_drive_automation(account_id: str, automation_id: str) -> bool:
    with _db() as c:
        cur = c.execute(
            "DELETE FROM drive_automations WHERE id=? AND account_id=?",
            (automation_id, account_id),
        )
    return cur.rowcount > 0


def _total_tokens(c: sqlite3.Connection, key_id: str) -> int:
    return int(c.execute(
        "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage WHERE key_id=?", (key_id,)
    ).fetchone()[0])


def _children_tokens(c: sqlite3.Connection, key_id: str) -> int:
    return int(c.execute(
        """SELECT COALESCE(SUM(ku.total_tokens),0)
           FROM key_usage ku JOIN api_keys child ON child.id=ku.key_id
           WHERE child.parent_key_id=?""", (key_id,)
    ).fetchone()[0])


def _period_tokens(c: sqlite3.Connection, row: sqlite3.Row, date_pattern: str) -> int:
    used = int(c.execute(
        "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage WHERE key_id=? AND date LIKE ?",
        (row["id"], date_pattern),
    ).fetchone()[0])
    if (row["key_type"] or "client") == "reseller":
        used += int(c.execute(
            """SELECT COALESCE(SUM(ku.total_tokens),0)
               FROM key_usage ku JOIN api_keys child ON child.id=ku.key_id
               WHERE child.parent_key_id=? AND ku.date LIKE ?""",
            (row["id"], date_pattern),
        ).fetchone()[0])
    return used


def daily_usage_fraction(app_name: str) -> float:
    """Fraction (0.0–1.0+) of today's DAILY token limit already used by this key,
    by display name. Used by the usage-window load shedding to decide who gets
    paused first under overload. Returns 0.0 when there's no daily limit or the
    key is unknown (so a missing limit never causes a pause). Cheap: one indexed
    SUM over today's key_usage."""
    today = _today_str()
    try:
        with _db() as c:
            row = c.execute(
                "SELECT id, daily_limit, key_type FROM api_keys WHERE name=?",
                (app_name,),
            ).fetchone()
            if not row:
                return 0.0
            limit = int(row["daily_limit"] or 0)
            if limit <= 0:
                return 0.0
            used = int(c.execute(
                "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage "
                "WHERE key_id=? AND date=?",
                (row["id"], today),
            ).fetchone()[0])
            if (row["key_type"] or "client") == "reseller":
                used += int(c.execute(
                    """SELECT COALESCE(SUM(ku.total_tokens),0)
                       FROM key_usage ku JOIN api_keys child ON child.id=ku.key_id
                       WHERE child.parent_key_id=? AND ku.date=?""",
                    (row["id"], today),
                ).fetchone()[0])
            return used / limit
    except Exception:
        return 0.0


def _remaining_tokens_for_row(c: sqlite3.Connection, row: sqlite3.Row,
                              seen: Optional[set[str]] = None) -> Optional[int]:
    seen = set(seen or ())
    if row["id"] in seen:
        return 0
    seen.add(row["id"])
    limit = int(row["token_limit"] or 0)
    remaining = None
    if limit > 0:
        used = _total_tokens(c, row["id"])
        if (row["key_type"] or "client") == "reseller":
            used += _children_tokens(c, row["id"])
        remaining = limit - used
    parent_id = row["parent_key_id"]
    if parent_id:
        parent = c.execute("SELECT * FROM api_keys WHERE id=?", (parent_id,)).fetchone()
        if not parent:
            return 0
        parent_remaining = _remaining_tokens_for_row(c, parent, seen)
        if parent_remaining is not None:
            remaining = parent_remaining if remaining is None else min(remaining, parent_remaining)
    return remaining


def _row_access_error(c: sqlite3.Connection, row: sqlite3.Row, now: Optional[float] = None,
                      today: Optional[str] = None, seen: Optional[set[str]] = None) -> Optional[str]:
    now = now or time.time()
    today = today or _today_str()
    seen = set(seen or ())
    if row["id"] in seen:
        return "disabled"
    seen.add(row["id"])
    if not row["active"]:
        return "disabled"
    if row["expires_at"] and row["expires_at"] <= now:
        return "expired"
    if row["daily_limit"] > 0 and _period_tokens(c, row, today) >= row["daily_limit"]:
        return "limit_exceeded"
    if row["monthly_limit"] > 0 and _period_tokens(c, row, today[:7] + "%") >= row["monthly_limit"]:
        return "limit_exceeded"
    remaining = _remaining_tokens_for_row(c, row)
    if remaining is not None and remaining <= 0:
        return "limit_exceeded"
    if row["parent_key_id"]:
        parent = c.execute("SELECT * FROM api_keys WHERE id=?", (row["parent_key_id"],)).fetchone()
        if not parent:
            return "disabled"
        return _row_access_error(c, parent, now=now, today=today, seen=seen)
    return None


def record_api_key_usage(key_id: Optional[str], input_tokens: int = 0,
                         output_tokens: int = 0, total_tokens: int = 0) -> None:
    """Registra consumo real depois da resposta do provider."""
    if not key_id:
        return
    input_tokens = max(0, int(input_tokens or 0))
    output_tokens = max(0, int(output_tokens or 0))
    total_tokens = max(0, int(total_tokens or input_tokens + output_tokens))
    today = _today_str()
    with _db() as c:
        c.execute(
            """INSERT INTO key_usage(key_id,date,request_count,input_tokens,output_tokens,total_tokens)
               VALUES(?,?,1,?,?,?)
               ON CONFLICT(key_id,date) DO UPDATE SET
                 request_count=request_count+1,
                 input_tokens=input_tokens+excluded.input_tokens,
                 output_tokens=output_tokens+excluded.output_tokens,
                 total_tokens=total_tokens+excluded.total_tokens""",
            (key_id, today, input_tokens, output_tokens, total_tokens),
        )
        c.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", (time.time(), key_id))


def create_reseller_key(name: str, token_limit: int, description: str = "",
                        validity_days: int = 0, notes: str = "") -> tuple[str, str]:
    return create_api_key(name, description=description, validity_days=validity_days,
                          notes=notes, key_type="reseller", token_limit=max(0, token_limit))


def create_reseller_client_key(reseller_id: str, name: str, token_limit: int,
                               daily_limit: int = 0, monthly_limit: int = 0,
                               validity_days: int = 0) -> tuple[str, str]:
    token_limit = max(0, int(token_limit))
    with _db() as c:
        reseller = c.execute("SELECT * FROM api_keys WHERE id=?", (reseller_id,)).fetchone()
        if not reseller or reseller["key_type"] != "reseller" or _row_access_error(c, reseller):
            raise ValueError("Revendedor inválido ou inativo.")
        available = _remaining_tokens_for_row(c, reseller)
        allocated = int(c.execute(
            "SELECT COALESCE(SUM(token_limit),0) FROM api_keys WHERE parent_key_id=?",
            (reseller_id,),
        ).fetchone()[0])
        pool = int(reseller["token_limit"] or 0)
        if token_limit <= 0:
            raise ValueError("Defina um limite de tokens maior que zero.")
        if pool > 0 and allocated + token_limit > pool:
            raise ValueError("O limite distribuído ultrapassa o saldo da chave mestre.")
        if available is not None and token_limit > available:
            raise ValueError("Saldo insuficiente na chave mestre.")
    return create_api_key(name, daily_limit=daily_limit, monthly_limit=monthly_limit,
                          validity_days=validity_days, key_type="client",
                          parent_key_id=reseller_id, token_limit=token_limit)


def list_reseller_clients(reseller_id: str) -> List[Dict[str, Any]]:
    return [k for k in list_api_keys() if k.get("parent_key_id") == reseller_id]


def create_reseller_session(raw_key: str) -> Optional[str]:
    verified = verify_api_key_db(raw_key)
    if not verified or verified.get("error") or verified.get("key_type") != "reseller":
        return None
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _db() as c:
        c.execute("DELETE FROM reseller_sessions WHERE expires_at<?", (now,))
        c.execute("INSERT INTO reseller_sessions(token,key_id,created_at,expires_at) VALUES(?,?,?,?)",
                  (token, verified["app_id"], now, now + SESSION_TTL))
    return token


def validate_reseller_session(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    with _db() as c:
        r = c.execute("""SELECT rs.key_id,rs.expires_at AS session_expires_at,ak.*
                         FROM reseller_sessions rs JOIN api_keys ak ON ak.id=rs.key_id
                         WHERE rs.token=?""", (token,)).fetchone()
        if not r or r["session_expires_at"] < time.time() or r["key_type"] != "reseller" or _row_access_error(c, r):
            c.execute("DELETE FROM reseller_sessions WHERE token=?", (token,))
            return None
        keys = {k["id"]: k for k in list_api_keys()}
        return keys.get(r["key_id"])


def delete_reseller_session(token: str) -> None:
    if token:
        with _db() as c:
            c.execute("DELETE FROM reseller_sessions WHERE token=?", (token,))


# ── Reseller accounts (email/password login + admin approval) ───────────────────
# A reseller account is a web_account with role='reseller'. It has no api_key while
# pending; on approval the admin creates a reseller master key and links it. The
# reseller logs in with email/password and never sees that key. Existing resellers
# (a reseller-type api_key without an account) keep working via the legacy master-key
# login until migrated; the admin can attach an account to such a key on approval.

def create_reseller_account(name: str, email: str, password: str) -> Dict[str, Any]:
    """Open signup: creates a PENDING reseller account.

    A reseller master key is created immediately but with balance=0 and INACTIVE,
    so web_accounts.key_id (FK, NOT NULL) is always valid. Approval just sets the
    balance and activates the key — until then the reseller can't distribute tokens.
    """
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if not name:
        raise ValueError("Informe seu nome.")
    if "@" not in email:
        raise ValueError("Informe um e-mail válido.")
    if len(password or "") < 8:
        raise ValueError("A senha precisa ter pelo menos 8 caracteres.")
    # Reject duplicate email up front (clearer than catching the FK/Integrity error).
    with _db() as c:
        if c.execute("SELECT 1 FROM web_accounts WHERE email=?", (email,)).fetchone():
            raise ValueError("Este e-mail já possui uma conta.")
    # Create the master key first (inactive, zero balance) to satisfy the FK.
    key_val, _reveal = create_reseller_key(
        name, token_limit=0, description=f"Revendedor (pendente): {email}"
    )
    v = verify_api_key_db(key_val)
    key_id = v["app_id"] if v else None
    if not key_id:
        raise ValueError("Falha ao criar a chave do revendedor.")
    account_id = str(uuid.uuid4()).replace("-", "")
    ph = _hash_pw(password)
    try:
        with _db() as c:
            # Deactivate the master key while pending.
            c.execute("UPDATE api_keys SET active=0 WHERE id=?", (key_id,))
            c.execute(
                """INSERT INTO web_accounts
                   (id,key_id,name,email,password_hash,password_salt,created_at,role,status)
                   VALUES(?,?,?,?,?,?,?,'reseller','pending')""",
                (account_id, key_id, name, email, ph["hash"], ph["salt"], time.time()),
            )
    except sqlite3.IntegrityError as exc:
        # Roll back the orphan key if the account insert failed.
        with _db() as c:
            c.execute("DELETE FROM api_keys WHERE id=?", (key_id,))
        if "email" in str(exc).lower():
            raise ValueError("Este e-mail já possui uma conta.") from exc
        raise ValueError("Não foi possível criar a conta.") from exc
    return {"account_id": account_id, "key_id": key_id, "status": "pending"}


def approve_reseller(account_id: str, token_limit: int,
                     validity_days: int = 0) -> Dict[str, Any]:
    """Admin approves a pending reseller: sets the balance and activates the key."""
    with _db() as c:
        acc = c.execute(
            "SELECT * FROM web_accounts WHERE id=? AND role='reseller'", (account_id,)
        ).fetchone()
        if not acc:
            raise ValueError("Conta de revendedor não encontrada.")
        key_id = acc["key_id"]
        expires = (time.time() + validity_days * 86400) if validity_days > 0 else None
        c.execute(
            "UPDATE api_keys SET token_limit=?, active=1, expires_at=?, updated_at=? WHERE id=?",
            (max(0, int(token_limit or 0)), expires, time.time(), key_id),
        )
        c.execute("UPDATE web_accounts SET status='active' WHERE id=?", (account_id,))
    return {"account_id": account_id, "key_id": key_id, "status": "active"}


def authenticate_reseller(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Login for reseller accounts. Returns account dict (with key_id) or raises."""
    email = (email or "").strip().lower()
    with _db() as c:
        acc = c.execute(
            "SELECT * FROM web_accounts WHERE email=? AND role='reseller'", (email,)
        ).fetchone()
    if not acc or not _verify_pw(password, acc["password_hash"], acc["password_salt"]):
        raise ValueError("E-mail ou senha incorretos.")
    if acc["status"] == "pending":
        raise ValueError("Sua conta ainda não foi aprovada pelo administrador.")
    if acc["status"] == "suspended":
        raise ValueError("Sua conta está suspensa. Contate o administrador.")
    return {"id": acc["id"], "key_id": acc["key_id"], "name": acc["name"],
            "email": acc["email"], "status": acc["status"]}


def get_reseller_account_by_id(account_id: str) -> Optional[Dict[str, Any]]:
    with _db() as c:
        acc = c.execute(
            "SELECT id,key_id,name,email,status,created_at FROM web_accounts WHERE id=? AND role='reseller'",
            (account_id,),
        ).fetchone()
        return dict(acc) if acc else None


def reseller_session_account(raw_token: str) -> Optional[Dict[str, Any]]:
    """Resolve a web-session token to an ACTIVE reseller account (with balance)."""
    account_id = get_web_account_id_by_session(raw_token)
    if not account_id:
        return None
    with _db() as c:
        acc = c.execute(
            "SELECT id,key_id,name,email,status FROM web_accounts WHERE id=? AND role='reseller'",
            (account_id,),
        ).fetchone()
    if not acc or acc["status"] != "active":
        return None
    key = next((k for k in list_api_keys() if k["id"] == acc["key_id"]), None)
    return {
        "id": acc["id"], "key_id": acc["key_id"], "name": acc["name"],
        "email": acc["email"], "status": acc["status"],
        "token_limit": int(key["token_limit"]) if key and key.get("token_limit") else 0,
        "tokens_remaining": key.get("tokens_remaining") if key else None,
    }


def set_reseller_status(account_id: str, status: str) -> None:
    if status not in ("active", "suspended", "pending"):
        raise ValueError("Status inválido.")
    with _db() as c:
        acc = c.execute("SELECT key_id FROM web_accounts WHERE id=? AND role='reseller'",
                        (account_id,)).fetchone()
        if not acc:
            raise ValueError("Conta não encontrada.")
        c.execute("UPDATE web_accounts SET status=? WHERE id=?", (status, account_id))
        # Suspending the reseller also deactivates its master key (and thus blocks
        # all its client keys via the balance/parent checks). Reactivating only
        # restores the key when leaving suspension (a pending account stays inactive).
        if acc["key_id"]:
            new_active = 0 if status in ("suspended", "pending") else 1
            c.execute("UPDATE api_keys SET active=? WHERE id=?", (new_active, acc["key_id"]))


def recharge_reseller(account_id: str, add_tokens: int) -> None:
    """Admin adds (or removes, if negative) tokens to the reseller's master balance."""
    with _db() as c:
        acc = c.execute("SELECT key_id FROM web_accounts WHERE id=? AND role='reseller'",
                        (account_id,)).fetchone()
        if not acc:
            raise ValueError("Revendedor não encontrado.")
        # ATOMIC: SQL-level MAX(0, current + delta). Previous version was
        # read-modify-write — two concurrent admin clicks at the same instant
        # would both read the same starting balance and apply +N once, losing
        # one increment. Doing it in a single UPDATE removes the race.
        c.execute(
            "UPDATE api_keys SET token_limit = MAX(0, COALESCE(token_limit, 0) + ?), "
            "updated_at=? WHERE id=?",
            (int(add_tokens), time.time(), acc["key_id"]),
        )


def delete_reseller_account(account_id: str) -> None:
    """Delete a reseller account and its master key (clients cascade via FK)."""
    with _db() as c:
        acc = c.execute("SELECT key_id FROM web_accounts WHERE id=? AND role='reseller'",
                        (account_id,)).fetchone()
        if not acc:
            return
        c.execute("DELETE FROM web_accounts WHERE id=?", (account_id,))
        if acc["key_id"]:
            # Deleting the master key cascades to client keys (parent_key_id FK).
            c.execute("DELETE FROM api_keys WHERE id=? OR parent_key_id=?",
                      (acc["key_id"], acc["key_id"]))


def list_reseller_accounts() -> List[Dict[str, Any]]:
    """List all reseller accounts with balance, status and client/usage summary."""
    keys = {k["id"]: k for k in list_api_keys()}
    out = []
    with _db() as c:
        rows = c.execute(
            "SELECT id,key_id,name,email,status,created_at FROM web_accounts WHERE role='reseller' ORDER BY created_at DESC"
        ).fetchall()
    for acc in rows:
        key = keys.get(acc["key_id"])
        clients = [k for k in keys.values() if k.get("parent_key_id") == acc["key_id"]]
        client_usage = sum(int(k.get("tokens_total") or 0) for k in clients)
        out.append({
            "id": acc["id"], "name": acc["name"], "email": acc["email"],
            "status": acc["status"], "created_at": acc["created_at"],
            "key_id": acc["key_id"] if key else None,
            "token_limit": int(key["token_limit"]) if key and key.get("token_limit") else 0,
            "tokens_remaining": key.get("tokens_remaining") if key else None,
            "clients_count": len(clients),
            "clients_usage": client_usage,
        })
    return out


def list_unlinked_reseller_keys() -> List[Dict[str, Any]]:
    """Legacy reseller master keys that have no web_account yet (for migration)."""
    linked = set()
    with _db() as c:
        for r in c.execute("SELECT key_id FROM web_accounts WHERE role='reseller'").fetchall():
            linked.add(r["key_id"])
    return [k for k in list_api_keys()
            if (k.get("key_type") == "reseller") and k["id"] not in linked]


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    today = _today_str()
    month = _month_str()
    with _db() as c:
        active = c.execute("SELECT COUNT(*) FROM api_keys WHERE active=1").fetchone()[0]
        today_r = c.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage WHERE date=?", (today,)).fetchone()[0]
        today_t = c.execute(
            "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage WHERE date=?", (today,)).fetchone()[0]
        month_r = c.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage WHERE date LIKE ?",
            (month + "%",)).fetchone()[0]
        month_t = c.execute(
            "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage WHERE date LIKE ?",
            (month + "%",)).fetchone()[0]
        total_r = c.execute(
            "SELECT COALESCE(SUM(request_count),0) FROM key_usage").fetchone()[0]
        total_t = c.execute(
            "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage").fetchone()[0]
        revenue = c.execute("""
            SELECT COALESCE(SUM(ku.total_tokens * ak.price_per_1k / 1000.0),0)
            FROM key_usage ku JOIN api_keys ak ON ku.key_id=ak.id
        """).fetchone()[0]
        resellers = c.execute(
            "SELECT COUNT(*) FROM api_keys WHERE key_type='reseller'").fetchone()[0]
    return {
        "active_keys": active, "today_requests": today_r,
        "month_requests": month_r, "total_requests": total_r,
        "today_tokens": today_t, "month_tokens": month_t, "total_tokens": total_t,
        "resellers": resellers,
        "total_revenue": round(revenue, 2), "date": today,
    }


def get_usage_chart(days: int = 14) -> List[Dict[str, Any]]:
    """Retorna uso diario dos ultimos N dias para o grafico."""
    result = []
    for i in range(days - 1, -1, -1):
        ts = time.time() - i * 86400
        d = _today_str(ts)
        with _db() as c:
            cnt = c.execute(
                "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage WHERE date=?", (d,)
            ).fetchone()[0]
        result.append({"date": d, "label": time.strftime("%d/%m", _local_struct(ts)), "count": cnt})
    return result


def get_key_usage_history(kid: str, days: int = 14) -> List[Dict[str, Any]]:
    result = []
    for i in range(days - 1, -1, -1):
        ts = time.time() - i * 86400
        d = _today_str(ts)
        with _db() as c:
            r = c.execute("SELECT total_tokens FROM key_usage WHERE key_id=? AND date=?",
                           (kid, d)).fetchone()
        result.append({"date": d, "label": time.strftime("%d/%m", _local_struct(ts)),
                       "count": r["total_tokens"] if r else 0})
    return result


def get_reseller_usage_history(kid: str, days: int = 14) -> List[Dict[str, Any]]:
    """Uso agregado da chave mestre e de todas as subchaves."""
    result = []
    for i in range(days - 1, -1, -1):
        ts = time.time() - i * 86400
        d = _today_str(ts)
        with _db() as c:
            total = c.execute(
                """SELECT COALESCE(SUM(ku.total_tokens),0)
                   FROM key_usage ku JOIN api_keys ak ON ak.id=ku.key_id
                   WHERE ku.date=? AND (ak.id=? OR ak.parent_key_id=?)""",
                (d, kid, kid),
            ).fetchone()[0]
        result.append({"date": d, "label": time.strftime("%d/%m", _local_struct(ts)),
                       "count": total})
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


# ── Dashboard overviews (premium dashboards) ────────────────────────────────────

def _usage_series_for_keys(key_ids: list, days: int) -> List[Dict[str, Any]]:
    """Daily total_tokens + request_count summed over the given key ids."""
    if not key_ids:
        return [
            {"date": _today_str(time.time() - i * 86400),
             "label": time.strftime("%d/%m", _local_struct(time.time() - i * 86400)),
             "tokens": 0, "requests": 0}
            for i in range(days - 1, -1, -1)
        ]
    placeholders = ",".join("?" * len(key_ids))
    out = []
    with _db() as c:
        for i in range(days - 1, -1, -1):
            ts = time.time() - i * 86400
            d = _today_str(ts)
            row = c.execute(
                f"""SELECT COALESCE(SUM(total_tokens),0) AS t, COALESCE(SUM(request_count),0) AS r
                    FROM key_usage WHERE date=? AND key_id IN ({placeholders})""",
                (d, *key_ids),
            ).fetchone()
            out.append({"date": d, "label": time.strftime("%d/%m", _local_struct(ts)),
                        "tokens": int(row["t"]), "requests": int(row["r"])})
    return out


def get_admin_overview(days: int = 14) -> Dict[str, Any]:
    """Global overview for the admin premium dashboard: KPIs, time series, and
    rankings by reseller and by client."""
    keys = list_api_keys()
    stats = get_stats()
    resellers = list_reseller_accounts()

    # KPIs
    tokens_sold = sum(int(a["token_limit"] or 0) for a in resellers)          # saldo distribuído aos revendedores
    tokens_in_circulation = sum(int(a["tokens_remaining"] or 0) for a in resellers if a["tokens_remaining"] is not None)
    active_resellers = sum(1 for a in resellers if a["status"] == "active")
    client_keys = [k for k in keys if (k.get("key_type") or "client") == "client"]
    active_clients = sum(1 for k in client_keys if k.get("active"))

    # Time series (all usage)
    series = _usage_series_for_keys([k["id"] for k in keys], days)

    # Ranking por revendedor (consumo dos clientes dele)
    reseller_rank = sorted(
        [{"name": a["name"], "email": a["email"], "status": a["status"],
          "tokens_used": int(a["clients_usage"] or 0),
          "tokens_remaining": a["tokens_remaining"],
          "clients": a["clients_count"]} for a in resellers],
        key=lambda x: x["tokens_used"], reverse=True,
    )[:10]

    # Ranking por cliente (todos)
    client_rank = sorted(
        [{"name": k["name"], "tokens_used": int(k.get("tokens_total") or 0),
          "tokens_remaining": k.get("tokens_remaining"),
          "active": bool(k.get("active"))} for k in client_keys],
        key=lambda x: x["tokens_used"], reverse=True,
    )[:10]

    return {
        "kpis": {
            "tokens_today": stats["today_tokens"],
            "tokens_month": stats["month_tokens"],
            "tokens_total": stats["total_tokens"],
            "requests_today": stats["today_requests"],
            "tokens_sold": tokens_sold,
            "tokens_in_circulation": tokens_in_circulation,
            "active_resellers": active_resellers,
            "active_clients": active_clients,
            "revenue": stats["total_revenue"],
        },
        "series": series,
        "reseller_rank": reseller_rank,
        "client_rank": client_rank,
        "days": days,
    }


def get_reseller_overview(reseller_key_id: str, days: int = 14) -> Dict[str, Any]:
    """Overview for a single reseller's premium dashboard: own balance, what was
    distributed/consumed, time series of the reseller's clients, and client ranking."""
    keys = {k["id"]: k for k in list_api_keys()}
    master = keys.get(reseller_key_id)
    clients = [k for k in keys.values() if k.get("parent_key_id") == reseller_key_id]

    distributed = sum(int(k.get("token_limit") or 0) for k in clients)
    consumed = sum(int(k.get("tokens_total") or 0) for k in clients)
    balance = master.get("tokens_remaining") if master else None

    series = _usage_series_for_keys([k["id"] for k in clients], days)

    client_rank = sorted(
        [{"name": k["name"], "preview": k.get("key_preview", ""),
          "tokens_used": int(k.get("tokens_total") or 0),
          "tokens_limit": int(k.get("token_limit") or 0),
          "tokens_remaining": k.get("tokens_remaining"),
          "active": bool(k.get("active"))} for k in clients],
        key=lambda x: x["tokens_used"], reverse=True,
    )

    return {
        "kpis": {
            "balance": balance,
            "token_limit": int(master["token_limit"]) if master and master.get("token_limit") else 0,
            "distributed": distributed,
            "consumed": consumed,
            "clients_count": len(clients),
            "active_clients": sum(1 for k in clients if k.get("active")),
        },
        "series": series,
        "client_rank": client_rank,
        "days": days,
    }


def get_client_overview(key_id: str, days: int = 14) -> Dict[str, Any]:
    """Overview for a single client's own key (the /cliente panel): balance,
    today/total usage, daily limit, and the time series of their consumption."""
    key = next((k for k in list_api_keys() if k["id"] == key_id), None)
    if not key:
        return {"kpis": {}, "series": _usage_series_for_keys([], days), "days": days}
    series = _usage_series_for_keys([key_id], days)
    return {
        "kpis": {
            "tokens_remaining": key.get("tokens_remaining"),
            "token_limit": int(key["token_limit"]) if key.get("token_limit") else 0,
            "tokens_today": int(key.get("tokens_today") or 0),
            "tokens_total": int(key.get("tokens_total") or 0),
            "daily_limit": int(key["daily_limit"]) if key.get("daily_limit") else 0,
        },
        "series": series,
        "plan": key.get("name", ""),
        "days": days,
    }


def attach_account_to_reseller_key(key_id: str, name: str, email: str, password: str) -> Dict[str, Any]:
    """Migration helper: give an existing (legacy) reseller master key an email/
    password account, preserving its balance and clients. After this the reseller
    can log in with email/password instead of pasting the master key."""
    name = (name or "").strip()
    email = (email or "").strip().lower()
    if not name:
        raise ValueError("Informe o nome do revendedor.")
    if "@" not in email:
        raise ValueError("Informe um e-mail válido.")
    if len(password or "") < 8:
        raise ValueError("A senha precisa ter pelo menos 8 caracteres.")
    with _db() as c:
        key = c.execute(
            "SELECT * FROM api_keys WHERE id=? AND key_type='reseller'", (key_id,)
        ).fetchone()
        if not key:
            raise ValueError("Chave de revendedor inválida.")
        if c.execute("SELECT 1 FROM web_accounts WHERE key_id=?", (key_id,)).fetchone():
            raise ValueError("Esta chave já tem uma conta vinculada.")
        if c.execute("SELECT 1 FROM web_accounts WHERE email=?", (email,)).fetchone():
            raise ValueError("Este e-mail já possui uma conta.")
        account_id = str(uuid.uuid4()).replace("-", "")
        ph = _hash_pw(password)
        c.execute(
            """INSERT INTO web_accounts
               (id,key_id,name,email,password_hash,password_salt,created_at,role,status)
               VALUES(?,?,?,?,?,?,?,'reseller','active')""",
            (account_id, key_id, name, email, ph["hash"], ph["salt"], time.time()),
        )
    return {"account_id": account_id, "key_id": key_id, "status": "active"}
