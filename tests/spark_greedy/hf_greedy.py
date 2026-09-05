import sys
sys.path.insert(0, '/tmp/spark_hf')
from transformers import AutoTokenizer
from configuration_spark import Spark2_5Config
from modeling_spark import Spark2_5ForCausalLM
import torch

path = '/tmp/spark_hf'
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
cfg = Spark2_5Config.from_pretrained(path)
model = Spark2_5ForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16)
model.eval()

# Match MLX generation: apply chat template? mlx_lm output was 'We need to understand the user's query'
# That looks like the model applied a chat template. Let's do plain prompt for parity first.
prompt = 'The Republic of Agents is'
ids = tok(prompt, return_tensors='pt').input_ids
print('prompt tokens:', ids.tolist())
out_ids = []
with torch.no_grad():
    cur = ids
    for _ in range(8):
        logits = model(cur).logits[0, -1]
        nxt = int(torch.argmax(logits))
        out_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
print('HF tokens:', out_ids)
print('HF text:', tok.decode(out_ids))
