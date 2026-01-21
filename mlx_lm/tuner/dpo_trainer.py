
import time
from functools import partial
from typing import List, Dict, Any, Optional
import math
from pathlib import Path
import json

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from .trainer import TrainingArgs, evaluate, grad_checkpoint, iterate_batches
from .utils import print_trainable_parameters, linear_to_lora_layers
from ..utils import load

class DPODataset:
    def __init__(self, data: List[Dict], tokenizer):
        self._data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        return self._data[idx]

    def process(self, sample):
        prompt = sample.get("prompt", "")
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        
        def get_text(content):
            if isinstance(content, list):
                return self.tokenizer.apply_chat_template(content, tokenize=False)
            return content

        if isinstance(chosen, list): 
             chosen_text = get_text(chosen)
             rejected_text = get_text(rejected)
             prompt_text = get_text(prompt) if prompt else ""
        else:
             chosen_text = prompt + chosen
             rejected_text = prompt + rejected
             prompt_text = prompt

        chosen_tokens = self.tokenizer.encode(chosen_text) 
        if self.tokenizer.eos_token_id is not None and (len(chosen_tokens) == 0 or chosen_tokens[-1] != self.tokenizer.eos_token_id):
             chosen_tokens.append(self.tokenizer.eos_token_id)
             
        rejected_tokens = self.tokenizer.encode(rejected_text)
        if self.tokenizer.eos_token_id is not None and (len(rejected_tokens) == 0 or rejected_tokens[-1] != self.tokenizer.eos_token_id):
             rejected_tokens.append(self.tokenizer.eos_token_id)
             
        prompt_tokens = self.tokenizer.encode(prompt_text)
        
        return (chosen_tokens, rejected_tokens, len(prompt_tokens))

def iterate_dpo_batches(dataset, batch_size, max_seq_length, loop=False, comm_group=None):
    indices = np.arange(len(dataset))
    if comm_group is not None:
        rank = comm_group.rank()
        world = comm_group.size()
        indices = indices[rank::world]

    while True:
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = [dataset.process(dataset[j]) for j in batch_indices]
            
            chosen = [x[0] for x in batch]
            rejected = [x[1] for x in batch]
            prompt_lens = [x[2] for x in batch]
            
            max_l = max(max(len(x) for x in chosen), max(len(x) for x in rejected))
            max_l = min(max_l, max_seq_length)
            
            pad_id = dataset.tokenizer.pad_token_id if dataset.tokenizer.pad_token_id is not None else dataset.tokenizer.eos_token_id
            if pad_id is None: pad_id = 0
            
            batch_c = np.full((len(batch), max_l), pad_id, dtype=np.int32)
            batch_r = np.full((len(batch), max_l), pad_id, dtype=np.int32)
            
            for j in range(len(batch)):
                 c = chosen[j][:max_l]
                 r = rejected[j][:max_l]
                 batch_c[j, :len(c)] = c
                 batch_r[j, :len(r)] = r
            
            yield mx.array(batch_c), mx.array(batch_r), mx.array(prompt_lens)
        
        if not loop:
            break

def compute_logprobs(logits, labels, prompt_lens):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    
    ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="none")
    log_probs = -ce
    
    B, L = shift_labels.shape
    indices = mx.arange(L)
    mask = indices[None, :] >= (prompt_lens[:, None] - 1)
    
    return (log_probs * mask).sum(axis=-1)

def dpo_loss_fn(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    
    logits = pi_logratios - ref_logratios
    losses = -nn.log_sigmoid(beta * logits)
    
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    
    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def train_dpo(
    model,
    ref_model,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    beta: float = 0.1,
):
    print(f"Starting DPO training..., iters: {args.iters}")
    
    ref_model.freeze()
    
    def loss_fn(model, batch_c, batch_r, prompt_lens):
        logits_c = model(batch_c)
        logits_r = model(batch_r)
        
        logp_c = compute_logprobs(logits_c, batch_c, prompt_lens)
        logp_r = compute_logprobs(logits_r, batch_r, prompt_lens)
        
        logits_ref_c = mx.stop_gradient(ref_model(batch_c))
        logits_ref_r = mx.stop_gradient(ref_model(batch_r))
        
        ref_logp_c = compute_logprobs(logits_ref_c, batch_c, prompt_lens)
        ref_logp_r = compute_logprobs(logits_ref_r, batch_r, prompt_lens)
        
        loss, reward_c, reward_r = dpo_loss_fn(logp_c, logp_r, ref_logp_c, ref_logp_r, beta=beta)
        return loss, reward_c, reward_r
        
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)
    
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch_c, batch_r, prompt_lens):
        (loss, rc, rr), grad = loss_value_and_grad(model, batch_c, batch_r, prompt_lens)
        optimizer.update(model, grad)
        return loss, rc, rr

    losses = []
    
    iterator = iterate_dpo_batches(train_dataset, args.batch_size, args.max_seq_length, loop=True)
    
    for it, (batch_c, batch_r, prompt_lens) in zip(range(1, args.iters + 1), iterator):
        loss, rc, rr = step(batch_c, batch_r, prompt_lens)
        mx.eval(state, loss, rc, rr)
        losses.append(loss.item())
        
        if it % args.steps_per_report == 0:
            avg_loss = np.mean(losses)
            print(f"Iter {it}: Loss {avg_loss:.4f}, Reward Chosen {rc.item():.4f}, Reward Rejected {rr.item():.4f}")
            losses = []
            
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_map(lambda x: x, model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            print(f"Saved adapters to {args.adapter_file}")

    adapter_weights = dict(tree_map(lambda x: x, model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print("Training Completed.")


def run_dpo(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    print(f"Loading model {args.model}")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    
    print("Loading reference model")
    # For DPO, simplest is to reload same model structure
    ref_model, _ = load(args.model, tokenizer_config={"trust_remote_code": True})
    ref_model.freeze()
    
    if args.resume_adapter_file:
         print(f"Resuming adapters from {args.resume_adapter_file}")
         model.load_weights(args.resume_adapter_file, strict=False)
         
    print("Applying LoRA")
    model.freeze()
    linear_to_lora_layers(
        model,
        args.num_layers,
        args.lora_parameters,
    )
    print_trainable_parameters(model)
    
    print(f"Loading DPO datasets from {args.data}")
    def load_dpo_data(path):
        if not path.exists(): return []
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        return DPODataset(data, tokenizer)

    data_path = Path(args.data)
    train_set = load_dpo_data(data_path / "train.jsonl")
    valid_set = load_dpo_data(data_path / "valid.jsonl")
    
    optimizer_config = args.optimizer_config[args.optimizer] if args.optimizer in args.optimizer_config else {}
    optimizer = optim.Adam(learning_rate=args.learning_rate, **optimizer_config)
    
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        steps_per_report=args.steps_per_report,
        steps_per_save=args.save_every,
        adapter_file=args.adapter_path,
        max_seq_length=args.max_seq_length
    )
    
    train_dpo(
        model=model,
        ref_model=ref_model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=training_args,
        beta=args.beta
    )
