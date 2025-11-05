# ABOUTME: Allows treating tests directory as a package for unittest targets.
# ABOUTME: Facilitates running isolated scheduler tests without pytest.

from tests.server_batched.util import ensure_mlx_stub

ensure_mlx_stub()
