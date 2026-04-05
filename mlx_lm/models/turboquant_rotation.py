"""Walsh-Hadamard Transform and random rotation for TurboQuant."""

import mlx.core as mx
import math


def walsh_hadamard_transform(x: mx.array) -> mx.array:
    """Fast Walsh-Hadamard Transform in MLX.

    O(d log d) butterfly operations. Input dimension must be power of 2.
    Operates on last dimension.

    Args:
        x: (..., d) where d is power of 2

    Returns:
        (..., d) transformed array, normalized by 1/sqrt(d)
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"Dimension must be power of 2, got {d}"

    h = 1
    while h < d:
        # Split into pairs at stride h
        x_reshaped = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        even = x_reshaped[..., 0, :]
        odd = x_reshaped[..., 1, :]
        # Butterfly: [a+b, a-b]
        new_even = even + odd
        new_odd = even - odd
        x = mx.stack([new_even, new_odd], axis=-2).reshape(*x.shape[:-1], d)
        h *= 2

    return x * (1.0 / math.sqrt(d))


def random_diagonal_sign(d: int, seed: int = 42) -> mx.array:
    """Random ±1 diagonal for randomized Hadamard transform.

    Args:
        d: dimension
        seed: random seed

    Returns:
        (d,) array of ±1 values
    """
    key = mx.random.key(seed)
    mask = mx.random.bernoulli(p=0.5, shape=(d,), key=key)
    return mx.where(mask, mx.array(1.0), mx.array(-1.0))


def randomized_hadamard_transform(x: mx.array, signs: mx.array) -> mx.array:
    """Randomized Hadamard Transform: WHT(diag(signs) @ x).

    This is the rotation used in PolarQuant. O(d log d).

    Args:
        x: (..., d)
        signs: (d,) random ±1 diagonal

    Returns:
        (..., d) rotated array
    """
    return walsh_hadamard_transform(x * signs)


def inverse_randomized_hadamard(x: mx.array, signs: mx.array) -> mx.array:
    """Inverse of randomized Hadamard transform.

    Since WHT is self-inverse (up to scaling) and diag(signs) is self-inverse:
    inverse = diag(signs) @ WHT(x)

    Args:
        x: (..., d)
        signs: (d,) same signs used in forward transform

    Returns:
        (..., d) inverse-rotated array
    """
    return walsh_hadamard_transform(x) * signs
