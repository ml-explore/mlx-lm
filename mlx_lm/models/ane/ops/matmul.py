import hashlib

import os

import tempfile
import torch

import numpy as np

try:
    import coremltools as ct

    from CoreML import MLModel, MLModelConfiguration
    import objc
except Exception as e:
    print(e)
    print("Please install CoreML, pyobjc and coremltools, see requirements.txt.")
    exit(1)


def ane_subgraph_builder(w : np.ndarray, b : np.ndarray = None, input_name="x", prefix : str = "") -> ct.models.MLModel:
    if w.ndim != 2:
        # reshape
        pass

    M, K = x.shape
    N = w.shape[0]

    output_name = f'{prefix}/out'

    input_features = [(input_name, ct.models.datatypes.Array(M, K))]
    output_features = [(output_name, ct.models.datatypes.Array(M, N))]

    # see https://apple.github.io/coremltools/v3.4/generated/coremltools.models.neural_network.builder.html
    builder = ct.models.neural_network.NeuralNetworkBuilder(input_features, output_features)
    builder.add_inner_product(name='matmul', input_name=input_name, output_name=output_name, W=weights, b=b, input_channels=K, output_channels=N, has_bias=b != None)

    spec = build.spec
    model = ct.models.MLModel(spec)
    return model, output_name


_cache = {}

# TODO (yiakwy) : add multi-levels cache
def _hash_matmul(W, b=None):
    algo = hashlib.sha256()
    algo.update(W.tobytes())
    if b is not None:
        algo.update(b.tobytes())

    return aglo.hexdigest()


def matmul(x : np.ndarray, w : np.ndarray, b : np.ndarray = None, prefix : str = "", input_name="x", model=None):

    key = _hash_matmul(W, b=b)

    if model is None:
        # TODO (yiakwy) : load from path

        cached = _cache.get(key, None)
        if cached is None or cached[2] is None:
            model, output_name = ane_subgraph_builder(w, b)

            # TODO (yiakwy) : save to path asynchronously
            modelproto_saved_path_dir = os.path.join(tempfile.gettempdir(), "mlx_ane_ops_cache")
            os.makedirs(modelproto_saved_path_dir, exist_ok=True)

            modelproto_saved_path = os.path.join(modelproto_saved_path_dir, f"matmul_{key}.mlmodel")

            compiled_mlmodel_path = MLModel.compileModelAtURL_error_(modelproto_saved_path, None)

            config = MLModelConfiguration.alloc().init()
            config.computeUnits = "all"
            mlmodel_obj = MLModel.modelWithContentsOfURL_configuration_error_(compiled_mlmodel_path, config, None)

            _cache[key] = (modelproto_saved_path, compiled_mlmodel_path, mlmodel_obj)

            model = mlmodel_obj
        else:
            model = cached[2]

    
    # get the moel


    inputs = {
        input_name : x
    }

    outputs = model.predictionFromFeatures_error_(inputs, None)

    out = np.array(outputs["out"], dtype=np.float32)
    return out


def test_fp8_group_scaled_gemm():
    test_configs = [
        (8, 32, 8)
        # (128, 256, 128)
        # (1024, 4096, 1024)
        # (4096, 16384, 4096),
    ]

    for M, K, N in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing M={M}, K={K}, N={N}")
        print(f"{'='*60}")

        torch.manual_seed(42)
        input_fp16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        weights = torch.randint(0, 2, (K, N), device="mps").float() * 2 - 1

        weights_fp16 = weights.half()

        # print(f"input_fp16 : {input_fp16}")
        # print(f"weights_fp16 : {weights_fp16}")

        assert(weights_fp16.is_contiguous())

        # correctness check
        output_torch = torch.matmul(input_fp16, weights_fp16)

        input_fp16_np = input_fp16.cpu().numpy()
        weights_fp16_np = weights_fp16.cpu().numpy()

        out = matmul(input_fp16_np, weights_fp16_np)

        max_error = np.max(np.abs(output_torch.cpu().numpy() - output_mlx))
        print(f"Max error between torch and mlx: {max_error:.6f}")
    pass


    if __name__ == "__main__":
        test_fp8_group_scaled_gemm()