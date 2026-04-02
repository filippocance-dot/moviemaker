from __future__ import annotations
import os
from passlib.context import CryptContext
from itsdangerous import URLSafeSerializer, BadSignature

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-in-production")
_pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")
_signer = URLSafeSerializer(SECRET_KEY, salt="session")

def hash_password(plain: str) -> str:
    return _pwd.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd.verify(plain, hashed)

def make_token(user_id: int) -> str:
    return _signer.dumps(user_id)

def decode_token(token: str) -> int | None:
    try:
        return _signer.loads(token)
    except BadSignature:
        return None
