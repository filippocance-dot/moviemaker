from __future__ import annotations
import os
import bcrypt
from itsdangerous import URLSafeSerializer, BadSignature

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
_signer = URLSafeSerializer(SECRET_KEY, salt="session")

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False

def make_token(user_id: int) -> str:
    return _signer.dumps(user_id)

def decode_token(token: str) -> int | None:
    try:
        return _signer.loads(token)
    except BadSignature:
        return None
