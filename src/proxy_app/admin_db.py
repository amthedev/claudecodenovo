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
            ("api_keys", "key_type",        "TEXT NOT NULL DEFAULT 'client'"),
            ("api_keys", "parent_key_id",   "TEXT"),
            ("api_keys", "token_limit",     "INTEGER NOT NULL DEFAULT 0"),
            ("key_usage", "input_tokens",   "INTEGER NOT NULL DEFAULT 0"),
            ("key_usage", "output_tokens",  "INTEGER NOT NULL DEFAULT 0"),
            ("key_usage", "total_tokens",   "INTEGER NOT NULL DEFAULT 0"),
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
    today = time.strftime("%Y-%m-%d", time.gmtime())
    month = time.strftime("%Y-%m", time.gmtime())
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
                "tokens_remaining": max(0, token_limit - effective_tokens) if token_limit else None,
                "allocated_tokens": allocated,
                "key_type": r["key_type"] or "client",
                "parent_key_id": r["parent_key_id"],
                "revenue_total": revenue,
            })
        return result


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
            if not parent or not parent["active"]:
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
            u = c.execute("SELECT total_tokens FROM key_usage WHERE key_id=? AND date=?",
                           (r["id"], today)).fetchone()
            if u and u["total_tokens"] >= dl:
                return {"error": "limit_exceeded"}
        month = today[:7]
        ml = r["monthly_limit"]
        if ml > 0:
            month_tokens = c.execute(
                "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage WHERE key_id=? AND date LIKE ?",
                (r["id"], month + "%"),
            ).fetchone()[0]
            if month_tokens >= ml:
                return {"error": "limit_exceeded"}
        remaining = _remaining_tokens_for_row(c, r)
        if remaining is not None and remaining <= 0:
            return {"error": "limit_exceeded"}
        return {"type": "managed", "app_id": r["id"], "app_name": r["name"],
                "key_preview": r["key_preview"], "key_type": r["key_type"] or "client"}


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


def _remaining_tokens_for_row(c: sqlite3.Connection, row: sqlite3.Row) -> Optional[int]:
    limit = int(row["token_limit"] or 0)
    if limit <= 0:
        return None
    used = _total_tokens(c, row["id"])
    if (row["key_type"] or "client") == "reseller":
        used += _children_tokens(c, row["id"])
    remaining = limit - used
    parent_id = row["parent_key_id"]
    if parent_id:
        parent = c.execute("SELECT * FROM api_keys WHERE id=?", (parent_id,)).fetchone()
        if parent:
            parent_remaining = _remaining_tokens_for_row(c, parent)
            if parent_remaining is not None:
                remaining = min(remaining, parent_remaining)
    return remaining


def record_api_key_usage(key_id: Optional[str], input_tokens: int = 0,
                         output_tokens: int = 0, total_tokens: int = 0) -> None:
    """Registra consumo real depois da resposta do provider."""
    if not key_id:
        return
    input_tokens = max(0, int(input_tokens or 0))
    output_tokens = max(0, int(output_tokens or 0))
    total_tokens = max(0, int(total_tokens or input_tokens + output_tokens))
    today = time.strftime("%Y-%m-%d", time.gmtime())
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
        if not reseller or reseller["key_type"] != "reseller" or not reseller["active"]:
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
        r = c.execute("""SELECT rs.key_id,rs.expires_at,ak.active,ak.key_type
                         FROM reseller_sessions rs JOIN api_keys ak ON ak.id=rs.key_id
                         WHERE rs.token=?""", (token,)).fetchone()
        if not r or r["expires_at"] < time.time() or not r["active"] or r["key_type"] != "reseller":
            return None
        keys = {k["id"]: k for k in list_api_keys()}
        return keys.get(r["key_id"])


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_stats() -> Dict[str, Any]:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    month = time.strftime("%Y-%m", time.gmtime())
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
        d = time.strftime("%Y-%m-%d", time.gmtime(ts))
        with _db() as c:
            cnt = c.execute(
                "SELECT COALESCE(SUM(total_tokens),0) FROM key_usage WHERE date=?", (d,)
            ).fetchone()[0]
        result.append({"date": d, "label": time.strftime("%d/%m", time.gmtime(ts)), "count": cnt})
    return result


def get_key_usage_history(kid: str, days: int = 14) -> List[Dict[str, Any]]:
    result = []
    for i in range(days - 1, -1, -1):
        ts = time.time() - i * 86400
        d = time.strftime("%Y-%m-%d", time.gmtime(ts))
        with _db() as c:
            r = c.execute("SELECT total_tokens FROM key_usage WHERE key_id=? AND date=?",
                           (kid, d)).fetchone()
        result.append({"date": d, "label": time.strftime("%d/%m", time.gmtime(ts)),
                       "count": r["total_tokens"] if r else 0})
    return result


def get_reseller_usage_history(kid: str, days: int = 14) -> List[Dict[str, Any]]:
    """Uso agregado da chave mestre e de todas as subchaves."""
    result = []
    for i in range(days - 1, -1, -1):
        ts = time.time() - i * 86400
        d = time.strftime("%Y-%m-%d", time.gmtime(ts))
        with _db() as c:
            total = c.execute(
                """SELECT COALESCE(SUM(ku.total_tokens),0)
                   FROM key_usage ku JOIN api_keys ak ON ak.id=ku.key_id
                   WHERE ku.date=? AND (ak.id=? OR ak.parent_key_id=?)""",
                (d, kid, kid),
            ).fetchone()[0]
        result.append({"date": d, "label": time.strftime("%d/%m", time.gmtime(ts)),
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
