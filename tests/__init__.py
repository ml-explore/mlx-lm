# ABOUTME: Allows treating tests directory as a package for unittest targets.
# ABOUTME: Facilitates running isolated scheduler tests without pytest.

try:
    import mlx  # noqa: F401
except Exception:
    from tests.server_batched.util import ensure_mlx_stub

    ensure_mlx_stub()
