"""Unit tests for the dependency-free auth module."""

from binderforge import auth


def test_password_hash_roundtrip():
    h = auth.hash_password("password123")
    assert h.startswith("pbkdf2_sha256$")
    assert auth.verify_password("password123", h) is True
    assert auth.verify_password("wrong-password", h) is False


def test_register_login_and_duplicate(tmp_path):
    store = auth.AuthStore(str(tmp_path / "users.db"))
    uid = store.create_user("a@b.com", "password123")
    assert store.verify_login("a@b.com", "password123") == uid

    try:
        store.verify_login("a@b.com", "nope")
        assert False, "wrong password should raise"
    except auth.AuthError:
        pass

    try:
        store.create_user("a@b.com", "another-pass")
        assert False, "duplicate email should raise"
    except auth.AuthError:
        pass


def test_short_password_rejected(tmp_path):
    store = auth.AuthStore(str(tmp_path / "users.db"))
    try:
        store.create_user("x@y.com", "short")
        assert False, "short password should raise"
    except auth.AuthError:
        pass


def test_token_issue_and_verify(tmp_path):
    secret = auth.load_secret(str(tmp_path))
    token = auth.issue_token(secret, "user-123")
    assert auth.verify_token(secret, token) == "user-123"
    assert auth.verify_token(secret, token + "x") is None
    assert auth.verify_token(secret, "garbage.token") is None
    assert auth.verify_token("wrong-secret", token) is None


def test_daily_quota(tmp_path):
    store = auth.AuthStore(str(tmp_path / "users.db"))
    uid = store.create_user("quota@b.com", "password123")
    limit = 2

    # First two charges pass.
    store.check_and_charge(uid, "job-1", limit=limit)
    store.check_and_charge(uid, "job-2", limit=limit)
    assert store.usage_today(uid) == 2

    # Third is rejected.
    try:
        store.check_and_charge(uid, "job-3", limit=limit)
        assert False, "quota should be exceeded"
    except auth.QuotaExceeded:
        pass

    # Recording the same job again is idempotent (no double-charge).
    store.record_usage(uid, "job-2")
    assert store.usage_today(uid) == 2
