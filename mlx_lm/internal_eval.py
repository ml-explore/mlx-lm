
import sys
import time
from typing import List, Optional, Union, Tuple
import numpy as np
import mlx.core as mx
import mlx.nn as nn

try:
    from lm_eval.api.model import LM
    from lm_eval.api.registry import register_model
    from lm_eval.utils import eval_logger
except ImportError:
    pass

from .utils import load, generate_step

class MLXLM(LM):
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        batch_size: int = 1,
        max_length: int = 2048,
        trust_remote_code: bool = False,
        adapter_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.model_path = model
        self.tokenizer_path = tokenizer if tokenizer else model
        self._batch_size = int(batch_size) if batch_size else 1
        self._max_length = max_length
        
        # Load model/tokenizer
        self.model, self.tokenizer = load(model, tokenizer_config={"trust_remote_code": trust_remote_code}, adapter_path=adapter_path)
        self.model.eval()

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        # requests: list of (context, continuation)
        results = []
        
        # Batch processing
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i : i + self.batch_size]
            
            # Prepare inputs
            # We need to compute P(continuation | context)
            # Input: context + continuation
            # Target: continuation
            
            contexts = [x[0] for x in batch]
            continuations = [x[1] for x in batch]
            
            # Encode
            # We need full sequence (context + continuation) and context only (to mask)
            
            full_encodings = [self.tokenizer.encode(c + cont) for c, cont in zip(contexts, continuations)]
            ctx_encodings = [self.tokenizer.encode(c) for c in contexts]
            
            # Pad
            max_len = max(len(x) for x in full_encodings)
            
            # Padding token?
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            if pad_id is None: pad_id = 0

            input_ids = np.full((len(batch), max_len), pad_id, dtype=np.int32)
            
            loss_masks = np.zeros((len(batch), max_len), dtype=np.bool_)
            
            for j, (full, ctx) in enumerate(zip(full_encodings, ctx_encodings)):
                input_ids[j, :len(full)] = full
                # Mask is 1 where we want to predict.
                # We predict from len(ctx)-1 (logit for first token of cont) to len(full)-2
                
                # Input: [A, B, C, D] (ctx=AB, cont=CD)
                # Labels: [B, C, D, ?]
                # Logits at indices:
                # 0 (A) -> predict B (ctx)
                # 1 (B) -> predict C (cont start)
                # 2 (C) -> predict D (cont end)
                
                # We want logprobs for C and D.
                # C is predicted by B (index len(ctx)-1)
                # D is predicted by C (index len(full)-2)
                
                start_idx = len(ctx) - 1
                end_idx = len(full) - 1
                loss_masks[j, start_idx:end_idx] = 1
                
            input_arr = mx.array(input_ids)
            
            # Forward
            logits = self.model(input_arr)
            
            # Logprobs
            # Shift logits and inputs
            shift_logits = logits[:, :-1, :]
            shift_labels = input_arr[:, 1:]
            
            # Get specific logprobs
            # CrossEntropy gives loss (-logprob)
            ce = nn.losses.cross_entropy(shift_logits, shift_labels, reduction="none")
            log_probs = -ce
            
            # Adjust masks (shifted)
            # original mask index i corresponds to input_ids[i] predicting input_ids[i+1]
            # shifted logits index i corresponds to prediction of input_ids[i+1]
            # So mask indices align with shift_logits indices directly 0..L-2
            
            mask_arr = mx.array(loss_masks[:, :-1]) 
            
            # Sum logprobs where mask is true
            masked_log_probs = log_probs * mask_arr
            sum_log_probs = masked_log_probs.sum(axis=-1)
            
            # Check for greediness (is_greedy)
            # greedy if argmax(logit) == target
            pred_ids = shift_logits.argmax(axis=-1)
            is_greedy = (pred_ids == shift_labels) & mask_arr
            # Check if ALL tokens in continuation were greedy?
            # lm-eval semantics: is_greedy usually just comparison of sum? 
            # Actually loglikelihood returns (logprob, is_greedy)
            # is_greedy: True if continuation is the greedy generation.
            
            # We need to check if match is perfect over the continuation range.
            # Count mask set bits
            lengths = mask_arr.sum(axis=-1)
            matches = is_greedy.sum(axis=-1)
            greedy_flags = (matches == lengths)
            
            for ls, g in zip(sum_log_probs, greedy_flags):
                results.append((ls.item(), g.item()))
                
        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        # Not strictly needed for basic tasks MMLU/Arc
        return super().loglikelihood_rolling(requests)

    def generate_until(self, requests) -> List[str]:
        # requests: list of (context, until_args)
        results = []
        for req in requests:
            context, args = req
            until = args.get("until", [])
            max_gen_toks = args.get("max_gen_toks", 256)
            temp = args.get("temperature", 0.0)
            
            # MLX generate
            # Basic generation loop
            
            # For efficiency we should batch, but generate_until args vary per request usually?
            # lm-eval often groups by args.
            
            input_ids = self.tokenizer.encode(context)
            input_arr = mx.array([input_ids])
            
            # Generate
            # We can use mlx_lm.utils.generate but we need to stop on 'until' strings.
            # mlx_lm doesn't support list of stop strings easily yet in high level API?
            # We can implement simple loop.
            
            # Setup cache
            prompt = mx.array(input_ids)
            
            tokens = []
            for (token, prob), _ in zip(
                generate_step(prompt, self.model, temp=temp),
                range(max_gen_toks)
            ):
                if token == self.tokenizer.eos_token_id:
                    break
                tokens.append(token)
                
                # Check stop conditions
                if len(tokens) > 0:
                    text = self.tokenizer.decode(tokens)
                    # Check if any until string is in text
                    stop = False
                    for term in until:
                        if term in text:
                            stop = True
                            # Truncate
                            text = text.split(term)[0]
                            break
                    if stop:
                        break
                        
            # Final text
            text = self.tokenizer.decode(tokens)
            # Post-processing stop (if not caught in loop because of token boundaries)
            for term in until:
                 if term in text:
                     text = text.split(term)[0]
                     
            results.append(text)
            
        return results
