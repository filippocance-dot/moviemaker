from __future__ import annotations
import os
import hashlib
from itsdangerous import URLSafeSerializer, BadSignature

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
_signer = URLSafeSerializer(SECRET_KEY, salt="session")

def hash_password(plain: str) -> str:
    # Simple PBKDF2 hash for development (use proper bcrypt in production)
    import hashlib
    import secrets
    salt = secrets.token_hex(16)
    key = hashlib.pbkdf2_hmac('sha256', plain.encode(), salt.encode(), 100000)
    return f"{salt}${key.hex()}"

def verify_password(plain: str, hashed: str) -> bool:
    import hashlib
    try:
        salt, key_hex = hashed.split('$')
        key = hashlib.pbkdf2_hmac('sha256', plain.encode(), salt.encode(), 100000)
        return key.hex() == key_hex
    except (ValueError, AttributeError):
        return False

def make_token(user_id: int) -> str:
    return _signer.dumps(user_id)

def decode_token(token: str) -> int | None:
    try:
        return _signer.loads(token)
    except BadSignature:
        return None
