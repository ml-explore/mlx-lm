from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("AntonV/mamba2-130m-hf")
# model = AutoModelForCausalLM.from_pretrained("AntonV/mamba2-130m-hf")

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")
# model = AutoModelForCausalLM.from_pretrained("mistralai/Mamba-Codestral-7B-v0.1")

input_ids = tokenizer("write the quick sort in C", return_tensors="pt")["input_ids"]
print(tokenizer("write the quick sort in C", return_tensors="pt"))

# out = model.generate(input_ids, max_new_tokens=10, temperature=0.8, do_sample=True)
# print(tokenizer.batch_decode(out))


"""
python -m mlx_lm.generate --model AntonV/mamba2-130m-hf --prompt "Hey," --temp 0.8
python -m mlx_lm.generate --model AntonV/mamba2-1.3b-hf  --prompt "Hey," --temp 0.8
python -m mlx_lm.generate --model mistralai/Mamba-Codestral-7B-v0.1 --prompt "write the quick sort in C" --temp 0.8

python -m mlx_lm.generate --model mistralai/Mamba-Codestral-7B-v0.1 --prompt "import torch.nn as nn
import torch.nn.functional as F

# two layer mlp cals using pytorch in python
def MLP(nn.Module"


python -m mlx_lm.lora \
--train --model AntonV/mamba2-130m-hf \
--data mlx-community/wikisql \
--num-layers -1 \
--iters 100 \
--batch-size 1 \
--steps-per-report 10 \
--max-seq-length 512 \
--adapter-path /Users/gokdenizgulmez/Desktop/mlx-lm/mlx_lm/tuner/test
"""