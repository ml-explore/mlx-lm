import mlx.core as mx
from mlx_lm import load
model, tok = load('/Users/dog/models/spark-x25-4b')
prompt = 'The Republic of Agents is'
ids = tok.encode(prompt)
print('prompt tokens:', ids)
cache = model.make_cache() if hasattr(model, 'make_cache') else None
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(model)
x = mx.array([ids])
logits = model(x, cache=cache)
out = []
for _ in range(8):
    tok_id = int(mx.argmax(logits[0, -1]).item())
    out.append(tok_id)
    logits = model(mx.array([[tok_id]]), cache=cache)
print('MLX tokens:', out)
print('MLX text:', tok.decode(out))
