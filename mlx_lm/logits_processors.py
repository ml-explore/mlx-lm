
import mlx.core as mx
import numpy as np
import logging

try:
    from outlines.fsm.guide import RegexGuide, JSONGuide
    from outlines.fsm.json_schema import build_regex_from_schema
except ImportError:
    logging.warning("Outlines not installed, structured output will not work.")

class OutlinesLogitsProcessor:
    def __init__(self, tokenizer, schema_str=None, regex_str=None):
        self.tokenizer = tokenizer
        
        # We need to ensure the tokenizer in Outlines is compatible. 
        # Outlines uses its own tokenizer wrapper usually.
        # But we can pass the underlying huggingface tokenizer if available.
        # mlx_lm uses a TokenizerWrapper. We need the real HF tokenizer.
        # self.tokenizer._tokenizer should be the HF tokenizer if it's the wrapper custom one.
        
        hf_tokenizer = tokenizer
        if hasattr(tokenizer, "_tokenizer"):
             hf_tokenizer = tokenizer._tokenizer

        # Outlines caching
        # We assume clean state per request.
        
        if schema_str:
            regex_str = build_regex_from_schema(schema_str)
        
        if regex_str:
            try:
                self.guide = RegexGuide(regex_str, hf_tokenizer)
                self.state = self.guide.get_initial_state()
            except Exception as e:
                logging.error(f"Failed to create Outlines Guide: {e}")
                self.guide = None
                self.state = None
        else:
            self.guide = None
            self.state = None
            
        self.processed_tokens = [] # Keep track as list of ints

    def __call__(self, tokens, logits):
        if not self.guide or self.state is None:
            return logits

        # tokens is mx.array containing history + input.
        # We need to sync to get values.
        # Optimization: track length to know how many new tokens to process.
        
        # Warning: synchronization here is a bottleneck but unavoidable for CPU FSM.
        target_len = tokens.size
        current_len = len(self.processed_tokens)
        
        if target_len > current_len:
            # New tokens available. 
            # We need to extract them.
            # Slice and eval.
            new_tokens_arr = tokens[current_len:]
            new_tokens = new_tokens_arr.tolist()
            
            for token_id in new_tokens:
                self.state = self.guide.get_next_state(self.state, token_id)
                self.processed_tokens.append(token_id)
        
        # Now get allowed tokens for the NEXT token
        allowed_tokens = self.guide.get_next_instruction(self.state).allowed_tokens
        
        # Create mask
        # allowed_tokens is a list of ints.
        n_vocab = logits.shape[-1]
        
        # We want to set -inf to everything EXCEPT allowed_tokens.
        # Doing this efficiently in MLX:
        # Create a mask of -inf, set indices to 0.
        
        # If all tokens allowed (rare in FSM), return logits.
        # Outlines RegexGuide doesn't seemingly expose "all allowed".
        
        # Optimization: reuse mask buffer if possible? 
        # For now, create fresh.
        
        # We can use numpy to create mask then convert to mx or use mx ops.
        # mx doesn't have advanced indexing (scatter) easily for this? 
        # mx.zeros_like(logits) ...
        
        # Approach:
        # bias = np.full(n_vocab, -1e9, dtype=np.float32)
        # bias[allowed_tokens] = 0.0
        # return logits + mx.array(bias)
        
        # Using numpy is fast enough for vocabulary size usually.
        
        bias = np.full(n_vocab, -1e9, dtype=np.float32)
        bias[list(allowed_tokens)] = 0.0
        
        return logits + mx.array(bias)
