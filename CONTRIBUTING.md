# Contributing to MLX LM

We want to make contributing to this project as easy and transparent as
possible.

## AI Usage Policy

AI-generated code is allowed. What is not allowed is submitting code you do not
understand. You are 100% responsible for every line, however it was produced,
and must explicitly disclose the manner in which AI was employed.

It is strictly prohibited to use AI to write your posts for you (bug reports,
feature requests, pull request descriptions, Github discussions, responding to
humans, ...).

## Adding New Models

When creating a pull request to add new models, the pull request template
[new_model.md](https://github.com/ml-explore/mlx-lm/blob/main/.github/PULL_REQUEST_TEMPLATE/new_model.md)
must be used.

Below are some tips to port LLMs available on Hugging Face to MLX.

Check if the model has weights in the
[safetensors](https://huggingface.co/docs/safetensors/index) format. If not
[follow instructions](https://huggingface.co/spaces/safetensors/convert) to
convert it.

After that, add the model file to the
[`mlx_lm/models`](https://github.com/ml-explore/mlx-lm/tree/main/mlx_lm/models)
directory. You can see other examples there. We recommend starting from a model
that is similar to the model you are porting.

Make sure the name of the new model file is the same as the `model_type` in the
`config.json`, for example
[starcoder2](https://huggingface.co/bigcode/starcoder2-7b/blob/main/config.json#L17).

To determine the model layer names, we suggest either:

- Refer to the Transformers implementation if you are familiar with the
  codebase.
- Load the model weights and check the weight names which will tell you about
  the model structure.
- Look at the names of the weights by inspecting `model.safetensors.index.json`
  in the Hugging Face repo.

To add LoRA support edit
[`mlx_lm/tuner/utils.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py#L27-L60)

Finally, add a test for the new modle type to the [model
tests](https://github.com/ml-explore/mlx-lm/blob/main/tests/test_models.py).

You can run the tests with:

```shell
python -m unittest discover tests/
```

## Pull Requests

- Search for existing pull requests first before creating one.
- Make sure new code is covered by tests. Add new tests if not, and confirm
  the new tests fail in the main branch.
- If performance may be impacted, run benchmarks for both the main branch and
  the pull request.
- When providing benchmarking results, include scripts and reproduction steps.
- Format the code with `uvx pre-commit run --all` before submitting a pull
  request. You can also install git hooks to run it automatically:

  ```shell
  pip install pre-commit
  pre-commit install
  ```

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing to mlx-lm, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
