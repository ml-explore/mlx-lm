# Spark-X2.5-4B real-weights greedy decode parity

Prompt: `The Republic of Agents is`
Tokens (both HF torch bf16 and MLX fp16 default):

    [259, 4718, 455, 7478, 259, 4335, 312, 275]
    ' a company that provides a service to the'

Match: 8/8 tokens identical.

To reproduce, install `XHToken/Spark-X2.5-4B` and run:

    python mlx_greedy.py   # in a venv with mlx-lm (from this fork)
    python hf_greedy.py    # in a venv with transformers==4.57.1 + torch

The two venvs are kept separate on M4 because mlx-lm requires transformers>=5,
while the reference `modeling_spark.py` requires transformers<5.
