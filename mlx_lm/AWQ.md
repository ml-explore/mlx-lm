# AWQ

MLX LM supports Activation-aware Weight Quantization (AWQ)[^1]. AWQ uses the
activations of the model on a representative dataset to tune the quantization
parameters.

To get started, first install the requirements:

```
pip install mlx-lm[awq]
```

Use `mlx_lm.awq` to run AWQ on a given model. For example:

```
mlx_lm.awq --model mistralai/Mistral-7B-Instruct-v0.3
```

The script can take anywhere form a few minutes to several hours to run
depending on the model size and the number of samples.

Some important options, along with their default values, are:

- `--mlx-path mlx_model`: The location to save the AWQ model.
- `--bits 4`: Precision of the quantization.
- `--num-samples 32`: Number of samples to use. Using more samples can lead to
  better results but takes longer.
- `--n-grid 10`: The granularity of the AWQ search. A larger grid can lead to
  better results but takes longer.

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
    --path mlx_model \
    --upload-repo mlx-community/Mistral-7B-Instruct-v0.3-4bit-AWQ
```

[^1]: Refer to the [paper](https://arxiv.org/abs/2306.00978)
and [github repository](https://github.com/mit-han-lab/llm-awq) for more
details.
