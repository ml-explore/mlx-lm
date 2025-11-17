import hashlib

import os

import tempfile
import torch

import numpy as np
import coremltools.models.datatypes as datatypes

try:
    import coremltools as ct

    from Cocoa import NSURL
    from CoreML import MLModel, MLModelConfiguration
    from CoreML import MLDictionaryFeatureProvider, MLFeatureValue
    # import objc
except Exception as e:
    print(e)
    print("Please install CoreML, pyobjc and coremltools, see requirements.txt.")
    exit(1)


def ane_subgraph_builder(x_desc : tuple, w : np.ndarray, b : np.ndarray = None, input_name="x", output_name="out", prefix : str = "") -> ct.models.MLModel:
    if w.ndim != 2:
        # reshape
        pass

    M, K = x_desc
    N, K = w.shape

    output_name = f'{prefix}/{output_name}'

    input_features = [(input_name, ct.models.datatypes.Array(M, K))]
    output_features = [(output_name, ct.models.datatypes.Array(M, N))]

    # see https://apple.github.io/coremltools/v3.4/generated/coremltools.models.neural_network.builder.html
    builder = ct.models.neural_network.NeuralNetworkBuilder(input_features, output_features, disable_rank5_shape_mapping=True)
    builder.add_inner_product(name='matmul', input_name=input_name, output_name=output_name, 
        W=w, b=b, input_channels=K, output_channels=N, has_bias=b != None)

    spec = builder.spec
    spec.description.predictedFeatureName = output_name

    # ct.utils.convert_double_to_float_multiarray_type(spec)

    model = ct.models.MLModel(spec)
    return model, output_name


_cache = {}

# TODO (yiakwy) : add multi-levels cache
def _hash_matmul(W, b=None):
    algo = hashlib.sha256()
    algo.update(W.tobytes())
    if b is not None:
        algo.update(b.tobytes())

    return algo.hexdigest()


def save_model_proto(model : ct.models.MLModel, saved_path : str, model_name : str):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path, exist_ok=True)

    model.save(saved_path)


def matmul(x : np.ndarray, w : np.ndarray, b : np.ndarray = None, prefix : str = "", input_name="x", output_name="out", model=None, **configs):

    key = _hash_matmul(w, b=b)

    if model is None:
        cached = _cache.get(key, None)
        if cached is None:

            model, output_name = ane_subgraph_builder(x.shape, w, b, input_name=input_name)
            print( f"model : {model}")

            dump = configs.get("dump", True)
            saved_path = configs.get("saved_path", os.path.join(tempfile.gettempdir(), "mlx_ane_ops_cache"))
            model_name = configs.get("model_name", f"op_matmul_{key}.mlmodel")

            # TODO (yiakwy) : save to path asynchronously
            if dump:
                save_model_proto(model, saved_path, model_name)

            _cache[key] = (model)
        else:
            model = cached[0]
            output_name = f'{prefix}/{output_name}'

    inputs = {input_name : x.astype(np.float32)}
    
    outputs = model.predict(inputs)

    out = np.array(outputs[output_name], dtype=np.float32)
    return out


def test_fp8_group_scaled_gemm():
    test_configs = [
        # (8, 32, 8)
        # (1, 1, 1)
        # (128, 256, 128)
        (1024, 4096, 1024)
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


        weights_fp16_t = torch.zeros((N, K), dtype=torch.float16, device="mps")
        weights_fp16_t[:] = weights_fp16.T[:]

        weights_fp16_np = weights_fp16_t.cpu().numpy()

        output_ane = matmul(input_fp16_np, weights_fp16_np)

        max_error = np.max(np.abs(output_torch.cpu().numpy() - output_ane))
        print(f"Max error between torch and mlx: {max_error:.6f}")


if __name__ == "__main__":
    test_fp8_group_scaled_gemm()