# Native OSCAR INT2 cache

The native cache is deliberately opt-in. Existing calls to
`make_prompt_cache(model)` continue to construct the model's ordinary cache.
To opt in, pass an `OscarConfig` (or a matching dictionary):

```python
from mlx_lm.models.cache import make_prompt_cache

cache = make_prompt_cache(
    model,
    oscar_config={
        "group_size": 128,
        "sink_tokens": 64,
        "recent_tokens": 256,
        "rotation_dir": "/path/to/oscar-rotations",
        "bounded_attention": True,
    },
)
```

`OscarKVCache` implements the native cache protocol (`state`, `meta_state`,
`update_and_fetch`, `trim`, `filter`, `extend`, `extract`, and `merge`). Its
state has sink, packed history, and recent tiers. Prompt-cache serialization
uses a fixed ten-entry state and restores rotations only after the caller
binds the calibrated artifacts with `bind_rotations`.

Offline rotations can be produced from captured K/V tensors:

```bash
mlx_lm.oscar_calibrate --samples calibration.safetensors --out rotations
```

The sample file must contain `keys` and `values` tensors shaped `[B,H,T,D]`
or a sequence of layer tensors. The output contains per-layer K/V rotation
safetensors and `metadata.json`.

The algorithm is attributed to FutureMLS-Lab, *OSCAR: Offline Spectral
Covariance-Aware Rotation for 2-bit KV Cache Quantization*, arXiv:2605.17757.
The implementation is authored for native `mlx-lm`. SGLang was consulted only
for behavioral/API comparison; no SGLang source was copied or adapted. This
provenance statement and any required NOTICE entry must be reviewed before
external redistribution.
