# ABOUTME: Provides helper utilities for scheduler integration tests.
# ABOUTME: Includes stubs for MLX modules when unavailable in test envs.

import sys
import types

import numpy as np

class _FakeArray(list):
    def tolist(self):
        return list(self)


from pathlib import Path


def ensure_mlx_stub():
    if "mlx" in sys.modules:
        return

    mlx_module = types.ModuleType("mlx")
    core_module = types.ModuleType("mlx.core")
    metal_module = types.ModuleType("mlx.metal")

    core_module.fast = types.SimpleNamespace()
    core_module.array = lambda data, *_, **__: np.array(data)
    core_module.eval = lambda *_, **__: None
    core_module.random = types.SimpleNamespace(seed=lambda *_: None, state=None)
    core_module.compile = lambda func, *_, **__: func
    core_module.default_device = lambda: None
    core_module.new_stream = lambda *_: object()
    core_module.logsumexp = lambda logits, axis=-1, keepdims=True: 0
    core_module.zeros = lambda shape, dtype=None: np.zeros(
        shape if isinstance(shape, tuple) else (shape,), dtype=dtype or float
    )
    core_module.stack = lambda arrays, axis=0: np.stack(arrays, axis=axis)
    core_module.concatenate = lambda arrays, axis=0: np.concatenate(arrays, axis=axis)
    core_module.transpose = lambda array, axes: np.transpose(array, axes)
    core_module.expand_dims = lambda array, axis=0: np.expand_dims(array, axis)
    core_module.int32 = np.int32
    core_module.argmax = lambda array, axis=-1: types.SimpleNamespace(item=lambda: 0)
    core_module.argpartition = lambda arr, kth, axis=-1: _FakeArray([0])
    core_module.argsort = lambda arr, axis=-1: _FakeArray([0])
    core_module.get_peak_memory = lambda: 0.0
    core_module.take_along_axis = lambda arr, idx, axis=-1: _FakeArray([0])
    core_module.put_along_axis = lambda arr, idx, values, axis=-1: _FakeArray([0])
    core_module.where = lambda *_, **__: _FakeArray([0])
    core_module.__version__ = "0.0.0"

    metal_module.device_info = lambda: {"architecture": "stub"}
    metal_module.is_available = lambda: False

    nn_module = types.ModuleType("mlx.nn")
    nn_module.Module = object
    utils_module = types.ModuleType("mlx.utils")
    utils_module.tree_map_with_path = lambda func, tree: tree
    utils_module.tree_map = lambda func, tree: tree
    utils_module.tree_flatten = lambda tree: ([], None)
    utils_module.tree_reduce = lambda func, tree, initializer=None: initializer
    utils_module.tree_unflatten = lambda flat, spec: flat

    core_module.metal = metal_module
    mlx_module.core = core_module
    mlx_module.metal = metal_module
    mlx_module.nn = nn_module
    mlx_module.utils = utils_module
    mlx_module.uint32 = int
    mlx_module.float32 = float
    mlx_module.__version__ = "0.0.0"
    mlx_module.__path__ = []

    sys.modules["mlx"] = mlx_module
    sys.modules["mlx.core"] = core_module
    sys.modules["mlx.metal"] = metal_module
    sys.modules["mlx.nn"] = nn_module
    sys.modules["mlx.utils"] = utils_module
    if "huggingface_hub" not in sys.modules:
        hf_module = types.ModuleType("huggingface_hub")
        hf_module.snapshot_download = lambda *_, **__: ""
        hf_module.HfFileSystem = object
        hf_module.scan_cache_dir = lambda **__: types.SimpleNamespace(repos=[])
        sys.modules["huggingface_hub"] = hf_module
    if "transformers" not in sys.modules:
        transformers_module = types.ModuleType("transformers")

        class _PreTrainedTokenizer:
            def __init__(self, *_, **__):
                pass

        transformers_module.PreTrainedTokenizer = _PreTrainedTokenizer
        transformers_module.PreTrainedTokenizerFast = _PreTrainedTokenizer

        class _AutoTokenizer:
            _registry = {}

            @classmethod
            def from_pretrained(cls, *_, **__):
                return _PreTrainedTokenizer()

            @classmethod
            def register(cls, name, fast_tokenizer_class=None, **kwargs):
                cls._registry[name] = (fast_tokenizer_class, kwargs)

        transformers_module.AutoTokenizer = _AutoTokenizer
        sys.modules["transformers"] = transformers_module

    if "mlx_lm.generate" not in sys.modules:
        gen_module = types.ModuleType("mlx_lm.generate")

        class _GenerationResponse:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class _BatchGenerator:
            def __init__(self, *_, **__):
                self.active_batch = types.SimpleNamespace(uids=[])
                self._next_responses = []
                self._uids = []

            def insert(self, prompts, *_args, **_kwargs):
                # Return deterministic uids for tests
                start = len(self._uids)
                ids = list(range(start, start + len(prompts)))
                self._uids.extend(ids)
                return ids

            def next(self):
                swap, self._next_responses = self._next_responses, []
                return swap

        gen_module.GenerationResponse = _GenerationResponse
        gen_module.BatchGenerator = _BatchGenerator
        gen_module.stream_generate = lambda *_, **__: []
        sys.modules["mlx_lm.generate"] = gen_module

    if "mlx_lm.models.cache" not in sys.modules:
        cache_module = types.ModuleType("mlx_lm.models.cache")

        class _StubKVCache:
            def __init__(self):
                self.keys = None
                self.values = None
                self.offset = 0

            @property
            def state(self):
                if self.keys is None or self.values is None:
                    zeros = np.zeros((1, 1, 0, 1), dtype=np.float32)
                    return zeros, zeros
                return self.keys, self.values

            @state.setter
            def state(self, value):
                self.keys, self.values = value
                if hasattr(self.keys, "shape"):
                    self.offset = self.keys.shape[2]
                else:
                    self.offset = 0

        class _StubBatchKVCache:
            def __init__(self, left_padding):
                self.left_padding = np.array(left_padding)
                self.keys = None
                self.values = None
                self.offsets = np.array([0] * len(left_padding))

            @property
            def state(self):
                keys = self.keys
                values = self.values
                if keys is None or values is None:
                    zeros = np.zeros((len(self.left_padding), 1, 0, 1), dtype=np.float32)
                    keys = zeros
                    values = zeros
                return keys, values, self.offsets, self.left_padding

            @state.setter
            def state(self, value):
                keys, values, offsets, left_padding = value
                self.keys = keys
                self.values = values
                self.offsets = np.array(
                    offsets.tolist() if hasattr(offsets, "tolist") else offsets
                )
                self.left_padding = np.array(
                    left_padding.tolist() if hasattr(left_padding, "tolist") else left_padding
                )

        class _StubCacheList:
            def __init__(self, *caches):
                self.caches = list(caches)

        cache_module.make_prompt_cache = lambda *_: [_StubKVCache()]
        cache_module.trim_prompt_cache = lambda *_: None
        cache_module.can_trim_prompt_cache = lambda *_: False
        cache_module.KVCache = _StubKVCache
        cache_module.BatchKVCache = _StubBatchKVCache
        cache_module.CacheList = _StubCacheList
        sys.modules["mlx_lm.models.cache"] = cache_module

    if "mlx_lm.sample_utils" not in sys.modules:
        sample_module = types.ModuleType("mlx_lm.sample_utils")

        def _sampler_factory(**_):
            return lambda logits: types.SimpleNamespace(item=lambda: 0)

        sample_module.make_sampler = _sampler_factory
        sample_module.make_logits_processors = lambda *_, **__: []
        sys.modules["mlx_lm.sample_utils"] = sample_module

    if "mlx_lm.utils" not in sys.modules:
        utils_stub = types.ModuleType("mlx_lm.utils")
        utils_stub.common_prefix_len = lambda a, b: 0
        utils_stub.load = lambda *_, **__: (None, None)
        sys.modules["mlx_lm.utils"] = utils_stub

    if "mlx_lm" not in sys.modules:
        try:
            import importlib

            importlib.import_module("mlx_lm")
        except Exception:
            pkg = types.ModuleType("mlx_lm")
            pkg.__path__ = [str(Path(__file__).resolve().parents[2] / "mlx_lm")]
            from importlib.machinery import ModuleSpec

            spec = ModuleSpec(name="mlx_lm", loader=None, is_package=True)
            spec.submodule_search_locations = pkg.__path__
            pkg.__spec__ = spec
            sys.modules["mlx_lm"] = pkg

    if "mlx_lm.server_batched" not in sys.modules:
        subpkg = types.ModuleType("mlx_lm.server_batched")
        subpkg.__path__ = [str(Path(__file__).resolve().parents[2] / "mlx_lm" / "server_batched")]
        from importlib.machinery import ModuleSpec

        spec = ModuleSpec(name="mlx_lm.server_batched", loader=None, is_package=True)
        spec.submodule_search_locations = subpkg.__path__
        subpkg.__spec__ = spec
        sys.modules["mlx_lm.server_batched"] = subpkg


ensure_mlx_stub()
