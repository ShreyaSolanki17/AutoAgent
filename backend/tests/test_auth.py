"""Auth tests — JWT lifecycle, password hashing.

Pure crypto tests.  No DB, no LLM.
"""

import unittest
from datetime import datetime, timedelta, timezone

import jwt

from backend.auth import (
    _create_token,
    _decode_token,
    _hash_password,
    _verify_password,
    SECRET_KEY,
    ALGORITHM,
    TOKEN_EXPIRE_HOURS,
)


class TestTokenLifecycle(unittest.TestCase):

    def test_create_and_decode_token(self):
        token = _create_token("user-123", "test@example.com")
        payload = _decode_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "user-123")
        self.assertEqual(payload["email"], "test@example.com")

    def test_token_has_exp_and_iat(self):
        token = _create_token("user-123", "test@example.com")
        payload = _decode_token(token)
        self.assertIn("exp", payload)
        self.assertIn("iat", payload)

    def test_expired_token_returns_none(self):
        payload = {
            "sub": "user-123",
            "email": "test@example.com",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
        result = _decode_token(token)
        self.assertIsNone(result)

    def test_tampered_token_returns_none(self):
        token = _create_token("user-123", "test@example.com")
        tampered = token[:-5] + "XXXXX"
        result = _decode_token(tampered)
        self.assertIsNone(result)

    def test_token_ttl_is_24_hours(self):
        self.assertEqual(TOKEN_EXPIRE_HOURS, 24)

    def test_token_expires_in_24_hours(self):
        token = _create_token("user-123", "test@example.com")
        payload = _decode_token(token)
        exp = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
        iat = datetime.fromtimestamp(payload["iat"], tz=timezone.utc)
        delta_hours = (exp - iat).total_seconds() / 3600
        self.assertAlmostEqual(delta_hours, 24, delta=0.1)


class TestPasswordHashing(unittest.TestCase):

    def test_hash_and_verify(self):
        hashed = _hash_password("mypassword123")
        self.assertTrue(_verify_password("mypassword123", hashed))

    def test_wrong_password_fails(self):
        hashed = _hash_password("mypassword123")
        self.assertFalse(_verify_password("wrong", hashed))

    def test_hash_is_not_plaintext(self):
        hashed = _hash_password("mypassword123")
        self.assertNotIn("mypassword123", hashed)

    def test_different_hashes_for_same_password(self):
        """bcrypt uses random salts, so same password → different hashes."""
        h1 = _hash_password("password")
        h2 = _hash_password("password")
        self.assertNotEqual(h1, h2)

    def test_empty_password_hashes(self):
        hashed = _hash_password("")
        self.assertTrue(_verify_password("", hashed))
        self.assertFalse(_verify_password("notempty", hashed))
