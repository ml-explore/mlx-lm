# Learned Quantization 

To reduce the quality loss from quantization MLX LM has two options:

- Distilled Weight Quantization (DWQ)
- Activation-aware Weight Quantization (AWQ)[^1].

Both DWQ and AWQ use an example dataset to tune parameters of the model. DWQ
fine-tunes non-quantized parameters (including quantization scales and biases)
using the non-quantized model as a teacher. AWQ scales and clips the weights
prior to quantization. The scaling and clipping values are found with a grid
search minimizing the distance from the quantized hidden activations to the
non-quantized hidden activations

To get started, first install the requirements:

```
pip install mlx-lm[lwq]
```

### DWQ

Use `mlx_lm.dwq` to run DWQ on a given model. For example:

```bash
mlx_lm.dwq --model mistralai/Mistral-7B-Instruct-v0.3
```

Some important options, along with their default values are:

- `--mlx-path mlx_model`: The location to save the DWQ model.
- `--bits 4`: Precision of the quantization.
- `--num-samples 1024`: Number of samples to use. Using more samples can lead to
  better results but takes longer.
- `--batch-size 8`: Use a smaller batch size to reduce the memory footprint.

For a full list of options run:

```bash
mlx_lm.dwq --help
```

### AWQ 

Use `mlx_lm.awq` to run AWQ on a given model. For example:

```bash
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

### Evaluate

Once the training script finishes, you can evaluate the quality of the model
on downstream tasks using `mlx_lm.evaluate`. For example:

```bash
mlx_lm.evaluate \
    --model mlx_model \
    --tasks winogrande boolq arc_challenge arc_easy hellaswag openbookqa piqa social_iqa                     
```

### Upload to Hugging Face

Use `mlx_lm.upload` to upload the quantized model to the Hugging Face Hub. For
example:

```bash
mlx_lm.upload \
    --path mlx_model \
    --upload-repo mlx-community/Mistral-7B-Instruct-v0.3-3bit-DWQ
```

[^1]: Refer to the [paper](https://arxiv.org/abs/2306.00978)
and [github repository](https://github.com/mit-han-lab/llm-awq) for more
details.
