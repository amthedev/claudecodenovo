# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Pure auth helpers: hashing, key generation, value parsers.

Extracted from main.py to keep it slimmer. These are stateless: no globals, no
DB calls — safe to import from anywhere. Functions that DO touch globals or DB
(verify_api_key, _verify_proxy_api_key_value, etc.) stay in main.py for now.
"""
import hashlib
import hmac
import secrets
import time
from typing import Any, Dict, Optional


PASSWORD_ITERATIONS = 260_000


def utc_day() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def hash_secret(secret_value: str, salt: Optional[str] = None) -> Dict[str, str]:
    """PBKDF2-SHA256 hash a secret. Returns {salt, hash} as hex strings."""
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        secret_value.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    ).hex()
    return {"salt": salt, "hash": digest}


def verify_secret(secret_value: str, stored: Dict[str, str]) -> bool:
    if not secret_value or not stored:
        return False
    candidate = hash_secret(secret_value, stored.get("salt"))
    return hmac.compare_digest(candidate["hash"], stored.get("hash", ""))


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def generate_proxy_key() -> str:
    return "sk-" + secrets.token_urlsafe(32).replace("_", "").replace("-", "")[:42]


def parse_non_negative_int(value: Any) -> int:
    """Used for daily_limit / validity_days — any junk becomes 0."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def expiry_from_days(validity_days: int) -> Optional[float]:
    if validity_days <= 0:
        return None
    return time.time() + (validity_days * 86400)


def timestamp_value(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_expired(value: Any) -> bool:
    ts = timestamp_value(value)
    return bool(ts and ts <= time.time())


def extract_bearer_token(auth: Optional[str]) -> Optional[str]:
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return auth.strip()
