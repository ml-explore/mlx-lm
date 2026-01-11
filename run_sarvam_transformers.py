
import sys
import torch
from transformers import AutoTokenizer
from mlx_lm.models.sarvam_moe_transformers import SarvamMoEForCausalLM, SarvamMoEConfig

def run_sarvam_transformers():
    if len(sys.argv) < 2:
        print("Usage: python run_sarvam_transformers.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]

    print(f"Loading model from {model_path}...")
    
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {model_path}, trying default implementation if applicable or erroring out: {e}")
        # Fallback to a common tokenizer if appropriate or re-raise
        # tokenizer = AutoTokenizer.from_pretrained("sarvamai/sarvam-2b-v0.5")
        raise e

    # 2. Load Model
    # We might need to manually register if not using trust_remote_code=True with auto_map
    # Since we have the class imported, we can load weights directly if the config matches
    
    # Check if config exists or load it
    try:
        config = SarvamMoEConfig.from_pretrained(model_path)
    except Exception as e:
        print(f"Could not load SarvamMoEConfig from {model_path}: {e}")
        raise e

    print("Initializing SarvamMoEForCausalLM...")
    # Using from_pretrained with the class directly
    # Note: If weights are sharded or in safetensors, from_pretrained handles it.
    # We use torch_dtype=torch.float16 or bfloat16 to avoid OOM if possible, though user said fp8 checkpoint?
    # Transformers doesn't natively support loading FP8 in generic models easily without bitsandbytes or specific support.
    # If the checkpoint IS fp8, it might need specific handling or just load it.
    # Assuming standard loading for now.
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    model = SarvamMoEForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.bfloat16, 
    )
    model.to(device)

    print("Model loaded. Generating text...")
    
    input_text = "What is the capital of India?"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7
        )
        
    print("Output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_sarvam_transformers()
