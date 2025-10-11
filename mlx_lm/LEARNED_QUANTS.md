# Learned Quantization 

To reduce the quality loss from quantization MLX LM has several options:

- Distilled Weight Quantization (DWQ)
- Activation-aware Weight Quantization (AWQ)[^1]
- Dynamic quantization
- GPT Quantization (GPTQ)[^2]

All methods use calibration data to tune parameters or hyper-parameters of the
model. DWQ fine-tunes non-quantized parameters (including quantization scales
and biases) using the non-quantized model as a teacher. AWQ scales and clips
the weights prior to quantization. Dynamic quantization estimates the
sensitivity of a model's outputs to each layer and uses a higher precision for
layers which have higher sensitivity. GPTQ finds quantized weights which
minimize the squared error of each layer's output given the provided input.

Dynamic quantization is the fastest to run. DWQ takes longer but typically
yields better results. You can also cascade methods. For example a dynamically
quantized model can be further refined with DWQ.

To get started, first install the requirements:

```
pip install "mlx-lm[train]"
```

### DWQ

Use `mlx_lm.dwq` to run DWQ on a given model. For example:

```bash
mlx_lm.dwq --model Qwen/Qwen3-0.6B
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

#### Tips

- DWQ works best distilling to lower precision, anywhere from 2-bit to 4-bit
  models.
- Distilling 16-bit precision to 8-bit and even 6-bit often doesn't work well.
  The loss starts out so low that it's difficult to reduce further.
- Decreasing the quantization group size (e.g. `--group-size 32`) doubles the
  number of tunable parameters and can work much better.
- If the loss is oscillating and not going down consistently, try reducing the
  learning rate. If it is decreasing but slowly, try increasing the learning
  rate.
- As a rule of thumb, lower precision can benefit from a higher learning rate
  since the loss starts out higher. Conversely, higher precision needs a lower
  learning rate.


#### Memory Use

A few options to reduce memory use for DWQ:

- Distill from an 8-bit model instead of a 16-bit model. The 8-bit
  models are usually as good as 16-bit precision models.
- Use a shorter maximum sequence length. The default is 2048. Using
  `--max-seq-length 512` reduces the memory and still gets good results.
- Use a smaller batch size, e.g. `--batch-size 1`

### Dynamic Quantization

Use `mlx_lm.dynamic_quant` to generate a dynamic quantization of given model.
For example:

```bash
mlx_lm.dynamic_quant --model Qwen/Qwen3-0.6B
```

The script will estimate the sensitivity for each quantizable layer in the
model. It will then quantize the model using higher precision (default 5 bits)
for the more sensitive layers and lower precision (default 4 bits) for the
rest. The script also saves a JSON file with each layer's sensitivities which
saves needing to compute it multiple times to make different precision quants
of the same model.

Some important options are:

- `--target-bpw`: The target bits-per-weight. For a given set of quantization
  parameters only certain ranges are possible. For example, with the default
  parameters a BPW in the range `[4.5, 5.5]` is achievable.
- `--sensitivities`: A path to a precomputed sensitivities file.
- `--low-bits`: The number of bits to use for the less sensitive layers.
- `--high-bits`: The number of bits to use for the more sensitive layers.

### AWQ 

Use `mlx_lm.awq` to run AWQ on a given model. For example:

```bash
mlx_lm.awq --model Qwen/Qwen3-0.6B
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

### GPTQ

Use `mlx_lm.gptq` to run GPTQ on a given model. For example:

```bash
mlx_lm.gptq --model Qwen/Qwen3-0.6B
```

The script can take anywhere from a few minutes to several hours depending on
the model size.

Some important options, along with their default values, are:

- `--mlx-path mlx_model`: The location to save the AWQ model.
- `--bits 4`: Precision of the quantization.


### Evaluate

Once the quantization training finishes, you can evaluate the quality of the
model on downstream tasks using `mlx_lm.evaluate`. For example:

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
details on AWQ.
[^2]: Refer to the [paper](https://arxiv.org/abs/2210.17323) for more details
on GPTQ.
