# AWQ

MLX LM supports Activation-aware Weight Quantization (AWQ)[^1]. AWQ uses the
activations of the model on a representative dataset to tune the quantization
parameters.

Use `mlx_lm.awq` to run AWQ on a given model. For example:
```
mlx_lm.awq --model mistralai/Mistral-7B-Instruct-v0.3
```

The script can take a while to run depending on the model size and the number
of samples. Once it's finished, the model will be saved in the current
directory under `mlx_model`.

Main default options are 
- --bits 3 (3bit quantization)
- --group-size 64
- --num-samples 32
- --n-grid 10 

For a full list of options run:

```bash
mlx_lm.awq --help
```

You can specify the quantization precision and group size, the number of
samples to use, the save path, and more. 

Once the script finishes, you can evaluate the quality of the model on
downstream tasks using `mlx_lm.evaluate`. For example:

```bash
mlx_lm.evaluate \
    --model mlx_model \
    --tasks winogrande boolq arc_challenge arc_easy hellaswag openbookqa piqa social_iqa                      
```

To upload the AWQ model to the Hugging Face Hub, run:

```bash
mlx_lm.upload \
    --model mlx_model \
    --upload-repo mlx-community/Mistral-7B-Instruct-v0.3-3bit-AWQ
```

[^1]: Refer to the [paper](https://arxiv.org/abs/2306.00978)
and [github repository](https://github.com/mit-han-lab/llm-awq) for more
details.
