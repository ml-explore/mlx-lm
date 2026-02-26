"""Test HuggingFace inference with the SarvamMoE checkpoint."""
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
# _dbg_log_path = "/rak/debug.log"
def log_debug(hypothesis_id, location, message, data):
    with open(_dbg_log_path, "a") as f:
        f.write(json.dumps({
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data
        }) + "\n")

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "ckpt-28232"
    
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    model.eval()
    
    log_debug("HF_CONFIG", "test_hf_inference.py", "model_config", {
        "num_experts": model.config.num_experts,
        "num_experts_per_tok": model.config.num_experts_per_tok,
        "routed_scaling_factor": model.config.routed_scaling_factor,
        "n_group": getattr(model.config, "n_group", None),
        "topk_group": getattr(model.config, "topk_group", None),
        "moe_router_enable_expert_bias": getattr(model.config, "moe_router_enable_expert_bias", None),
    })
    
    moe_layer = model.model.layers[1].mlp
    log_debug("HF_GATE", "test_hf_inference.py", "gate_weight_stats", {
        "gate_weight_shape": list(moe_layer.gate.weight.shape),
        "gate_weight_norm": float(moe_layer.gate.weight.norm()),
        "gate_weight_min": float(moe_layer.gate.weight.min()),
        "gate_weight_max": float(moe_layer.gate.weight.max()),
        "expert_bias_shape": list(moe_layer.gate.expert_bias.shape),
        "expert_bias_min": float(moe_layer.gate.expert_bias.min()),
        "expert_bias_max": float(moe_layer.gate.expert_bias.max()),
    })
    
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "def fibonacci(n):",
    ]
    
    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print(f"{'='*50}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,  
                pad_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated_text}")
        
        log_debug("HF_OUTPUT", "test_hf_inference.py", "generation_result", {
            "prompt": prompt,
            "generated": generated_text[:200],  
            "input_ids": inputs["input_ids"][0].tolist(),
            "output_ids": outputs[0].tolist()[:20],  
        })

if __name__ == "__main__":
    main()

