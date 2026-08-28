"""User accounts and sessions for the web layer.

Deliberately dependency-free (sqlite3 + hashlib + hmac from the standard
library) so the web UI can gate access without pulling in a web framework stack
or a password-crypto wheel. Passwords are hashed with PBKDF2-HMAC-SHA256 and a
per-user salt; sessions are stateless HMAC-signed tokens carrying an expiry.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from typing import Optional, Tuple

_PBKDF2_ITERATIONS = 200_000
_TOKEN_TTL_SECONDS = 7 * 24 * 3600  # 7 days
FREE_DAILY_LIMIT = 2  # free jobs per user per Beijing calendar day
_BEIJING_OFFSET = 8 * 3600  # UTC+8


class AuthError(Exception):
    """Raised for bad credentials, duplicate emails, or invalid tokens."""


class QuotaExceeded(Exception):
    """Raised when a user has hit their free daily limit."""


def _now() -> int:
    return int(time.time())


def _beijing_date(ts: int) -> str:
    """Return the Beijing (UTC+8) calendar date 'YYYY-MM-DD' for a unix ts."""
    return time.strftime("%Y-%m-%d", time.gmtime(ts + _BEIJING_OFFSET))


# ── SQLite storage ────────────────────────────────────────────────────────
class AuthStore:
    """Thin SQLite wrapper for users."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_usage_user_date ON usage (user_id, date)"
            )

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_user(self, email: str, password: str) -> str:
        email = email.strip().lower()
        if len(password) < 8:
            raise AuthError("密码至少 8 位 / Password must be at least 8 characters")
        user_id = secrets.token_hex(16)
        password_hash = hash_password(password)
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO users (id, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (user_id, email, password_hash, _now()),
                )
        except sqlite3.IntegrityError:
            raise AuthError("该邮箱已注册 / This email is already registered")
        return user_id

    def verify_login(self, email: str, password: str) -> str:
        email = email.strip().lower()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, password_hash FROM users WHERE email = ?", (email,)
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            raise AuthError("邮箱或密码错误 / Incorrect email or password")
        return row["id"]

    def get_user(self, user_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, email, created_at FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── daily usage / quota ────────────────────────────────────────────
    def usage_today(self, user_id: str) -> int:
        day = _beijing_date(_now())
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM usage WHERE user_id = ? AND date = ?",
                (user_id, day),
            ).fetchone()
        return int(row["n"])

    def record_usage(self, user_id: str, job_id: str) -> None:
        """Charge one task against the user's daily quota (idempotent per job)."""
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM usage WHERE job_id = ?", (job_id,)
            ).fetchone()
            if exists:
                return
            conn.execute(
                "INSERT INTO usage (user_id, date, job_id, created_at) VALUES (?, ?, ?, ?)",
                (user_id, _beijing_date(_now()), job_id, _now()),
            )

    def check_and_charge(self, user_id: str, job_id: str, limit: int = FREE_DAILY_LIMIT) -> None:
        """Charge a task, raising QuotaExceeded if the free daily limit is hit.

        Paid users are not implemented yet; `limit` is the hook where a paid
        tier would raise or bypass the cap.
        """
        used = self.usage_today(user_id)
        if used >= limit:
            raise QuotaExceeded(
                f"今日免费任务已用完（{used}/{limit}）。付费解锁更多任务（即将上线）。"
                f" Daily free limit reached ({used}/{limit}); paid tier coming soon."
            )
        self.record_usage(user_id, job_id)


# ── Password hashing (PBKDF2) ─────────────────────────────────────────────
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${_b64(salt)}${_b64(dk)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iters, salt_b64, dk_b64 = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        salt = _unb64(salt_b64)
        expected = _unb64(dk_b64)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iters))
        return hmac.compare_digest(dk, expected)
    except Exception:  # noqa: BLE001 — malformed stored hash -> not a match
        return False


# ── Signed session tokens (stateless) ─────────────────────────────────────
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def issue_token(secret: str, user_id: str, ttl: int = _TOKEN_TTL_SECONDS) -> str:
    payload = {"uid": user_id, "exp": _now() + ttl}
    body = _b64(json.dumps(payload).encode("utf-8"))
    sig = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
    return body + "." + _b64(sig)


def verify_token(secret: str, token: str) -> Optional[str]:
    """Return the user_id if the token is valid and unexpired, else None."""
    try:
        body, sig = token.split(".", 1)
        expected = hmac.new(secret.encode("utf-8"), body.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64(expected), sig):
            return None
        payload = json.loads(_unb64(body).decode("utf-8"))
        if payload.get("exp", 0) < _now():
            return None
        return payload.get("uid")
    except Exception:  # noqa: BLE001 — anything malformed is simply invalid
        return None


def load_secret(data_dir: str) -> str:
    """Persist a signing secret so sessions survive restarts."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "auth_secret")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    secret = secrets.token_hex(32)
    with open(path, "w", encoding="utf-8") as f:
        f.write(secret)
    return secret
