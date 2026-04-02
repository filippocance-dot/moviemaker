from web.auth import hash_password, verify_password, make_token, decode_token

def test_hash_and_verify():
    h = hash_password("mysecret")
    assert verify_password("mysecret", h) is True
    assert verify_password("wrong", h) is False

def test_token_roundtrip():
    token = make_token(42)
    user_id = decode_token(token)
    assert user_id == 42

def test_token_invalid():
    assert decode_token("notavalidtoken") is None
