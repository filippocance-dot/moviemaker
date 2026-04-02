"""
conftest.py — shared pytest fixtures and hooks.

The test_routes module uses a shared TestClient that accumulates cookies
across tests. The autouse fixture below clears cookies before each test
so that auth-required routes are properly rejected when no session cookie
should be present.
"""
import pytest


@pytest.fixture(autouse=True)
def clear_test_client_cookies(request):
    """Clear cookies on the shared TestClient between tests in test_routes."""
    # Only act on tests inside test_routes
    if request.module.__name__.endswith("test_routes"):
        # Import lazily to avoid circular issues at collection time
        import tests.test_routes as tr
        tr.client.cookies.clear()
    yield
