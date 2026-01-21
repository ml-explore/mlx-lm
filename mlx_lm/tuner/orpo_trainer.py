
import time
from functools import partial
from typing import List, Dict, Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

from .trainer import TrainingArgs, evaluate, iterate_batches
from .dpo_trainer import DPODataset, iterate_dpo_batches, compute_logprobs
from .utils import print_trainable_parameters

def orpo_loss_fn(policy_chosen_logps, policy_rejected_logps, beta=0.1):
    # ORPO Loss = L_SFT + beta * L_OR
    # L_SFT is negative log prob of chosen
    
    # In DPO trainer, compute_logprobs returns sum of logprobs for the sequence.
    # To get "average logprob" we might need lengths. 
    # Usually SFT loss is mean over tokens.
    
    # We need n_tokens for chosen to normalize SFT correctly?
    # compute_logprobs in dpo_trainer returns summed log probs per sample.
    
    # We need to adapt compute_logprobs or get lengths.
    # For now, let's assume we can get basic SFT from policy_chosen_logps if normalized?
    # No, typically we compute CrossEntropy mean for SFT.
    
    # ORPO paper: 
    # L_OR = -log(sigma(log(odds_chosen) - log(odds_rejected)))
    # log(odds) = log(p/(1-p)) = log(p) - log(1-p)
    
    # We have log_p.
    # log(1-p) = log(1 - exp(log_p)) = log(-expm1(log_p)) ? No, 1-p is probability of NOT generating the token?
    # ORPO is typically per-sample level or per-token?
    # ORPO: "We calculate the odds ratio of the chosen response to the rejected response."
    
    # odds(y|x) = P(y|x) / (1 - P(y|x))
    # where P(y|x) is the likelihood of the sequence.
    
    # So we use the summed log probs.
    
    log_odds_chosen = policy_chosen_logps - mx.log1p(-mx.exp(policy_chosen_logps))
    log_odds_rejected = policy_rejected_logps - mx.log1p(-mx.exp(policy_rejected_logps))
    
    # Numerical stability issues here if p -> 1.
    # If p is small (typical for sequence likelihood), 1-p ~ 1, log(1-p) ~ 0.
    
    ratio = log_odds_chosen - log_odds_rejected
    l_or = -nn.log_sigmoid(ratio).mean()
    
    # SFT Loss
    # We want standard NLL per token.
    # policy_chosen_logps is SUM of log probs.
    # We typically need to divide by length to get NLL used for reporting, but for optimization we use sum?
    # Standard SFT trainer uses mean reduction.
    # If we optimize mean SFT + lambda * mean ORPO.
    
    l_sft = -policy_chosen_logps.mean() # This is sum of log probs. 
    # If we want per-token sft, we need to divide by total tokens in batch (or per sample length)
    
    # The magnitude of SFT (sum) vs ORPO (sigmoid) is different. 
    # SFT is ~ -Length * Entropy.
    
    # Let's check ORPO implementations (e.g. TRL). 
    # SFT is computed as NLL per token usually?
    # TRL: losses = policy_nll_loss + beta * odds_ratio_loss
    # policy_nll_loss = -logprobs / mask.sum() -> Mean NLL per token.
    
    return l_sft, l_or

def train_orpo(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    beta: float = 0.1,
):
    print(f"Starting ORPO training..., iters: {args.iters}")
    
    def loss_fn(model, batch_c, batch_r, prompt_lens):
        logits_c = model(batch_c)
        logits_r = model(batch_r)
        
        # We need per-token logprobs for SFT part to normalize properly?
        # compute_logprobs returns SUM.
        
        # Let's modify logic or reuse compute_logprobs to get raw CE
        # Re-implement log-probs retrieval locally for flexibility
        
        # Chosen
        shift_logits_c = logits_c[:, :-1, :]
        shift_labels_c = batch_c[:, 1:]
        ce_c = nn.losses.cross_entropy(shift_logits_c, shift_labels_c, reduction="none")
        
        # Masking chosen
        L_c = shift_labels_c.shape[1]
        indices_c = mx.arange(L_c)
        mask_c = indices_c[None, :] >= (prompt_lens[:, None] - 1)
        # remove padding impact if any (pad_id=0 usually)
        if hasattr(train_dataset.tokenizer, "pad_token_id") and train_dataset.tokenizer.pad_token_id is not None:
             mask_c = mask_c & (shift_labels_c != train_dataset.tokenizer.pad_token_id)
        
        log_prob_c = (-ce_c * mask_c).sum(axis=-1)
        n_tokens_c = mask_c.sum()
        
        # SFT Loss = -log_prob / n_tokens
        l_sft = -log_prob_c.sum() / (n_tokens_c + 1e-9)
        
        # Rejected
        shift_logits_r = logits_r[:, :-1, :]
        shift_labels_r = batch_r[:, 1:]
        ce_r = nn.losses.cross_entropy(shift_logits_r, shift_labels_r, reduction="none")
        
        L_r = shift_labels_r.shape[1]
        indices_r = mx.arange(L_r)
        mask_r = indices_r[None, :] >= (prompt_lens[:, None] - 1)
        if hasattr(train_dataset.tokenizer, "pad_token_id") and train_dataset.tokenizer.pad_token_id is not None:
             mask_r = mask_r & (shift_labels_r != train_dataset.tokenizer.pad_token_id)

        log_prob_r = (-ce_r * mask_r).sum(axis=-1)
        
        # ORPO Loss
        # log_odds = log_prob - log(1 - exp(log_prob))
        # log1p(-exp(x))
        log_odds_c = log_prob_c - mx.log1p(-mx.exp(log_prob_c) + 1e-9)
        log_odds_r = log_prob_r - mx.log1p(-mx.exp(log_prob_r) + 1e-9)
        
        ratio = log_odds_c - log_odds_r
        l_or = -nn.log_sigmoid(ratio).mean()
        
        total_loss = l_sft + beta * l_or
        
        return total_loss, l_sft, l_or
        
    loss_value_and_grad = nn.value_and_grad(model, loss_fn)
    
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(batch_c, batch_r, prompt_lens):
        (loss, sft, or_loss), grad = loss_value_and_grad(model, batch_c, batch_r, prompt_lens)
        optimizer.update(model, grad)
        return loss, sft, or_loss

    losses = []
    
    iterator = iterate_dpo_batches(train_dataset, args.batch_size, args.max_seq_length, loop=True)
    
    for it, (batch_c, batch_r, prompt_lens) in zip(range(1, args.iters + 1), iterator):
        loss, sft, or_loss = step(batch_c, batch_r, prompt_lens)
        mx.eval(state, loss, sft, or_loss)
        losses.append(loss.item())
        
        if it % args.steps_per_report == 0:
            avg_loss = np.mean(losses)
            print(f"Iter {it}: Loss {avg_loss:.4f}, SFT {sft.item():.4f}, ORPO {or_loss.item():.4f}")
            losses = []
            
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_map(lambda x: x, model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            print(f"Saved adapters to {args.adapter_file}")

    adapter_weights = dict(tree_map(lambda x: x, model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print("Training Completed.")

def run_orpo(args):
    # Same as run_dpo but calling train_orpo and no ref_model
    import mlx.core as mx
    from .utils import load, linear_to_lora_layers, print_trainable_parameters
    
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    print(f"Loading model {args.model}")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    
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
    
    def load_dpo_data(path):
        import json
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
    
    train_orpo(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=valid_set,
        args=training_args,
        beta=args.beta
    )
