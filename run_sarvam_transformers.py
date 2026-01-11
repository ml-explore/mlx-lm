
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def run_sarvam_transformers():
    if len(sys.argv) < 2:
        print("Usage: python run_sarvam_transformers.py <model_path>")
        sys.exit(1)
        
    model_path = sys.argv[1]

    print(f"Loading model from {model_path}...")
    
    import time
    import os
    
    # Check for incomplete downloads
    for root, dirs, files in os.walk(model_path):
        for file in files:
            if file.endswith(".gstmp"):
                print(f"WARNING: Found temporary file {file}. Your download might be incomplete.")

    t0 = time.time()
    # 1. Load Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Failed to load tokenizer from {model_path}: {e}")
        raise e
    print(f"Tokenizer loaded in {time.time() - t0:.2f}s")

    # 2. Load Model
    t1 = time.time()
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Could not load AutoConfig from {model_path}: {e}")
        raise e

    print("Initializing AutoModelForCausalLM...")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t1:.2f}s")

    for name, param in model.named_parameters():
        print(name, type(param), param.device)
        break

    print("Model loaded. Running single forward pass...")
    test_input = tokenizer("Hello", return_tensors="pt").to(device)
    with torch.no_grad():
        t_test = time.time()
        out = model(**test_input)
        print(f"Forward pass successful in {time.time() - t_test:.2f}s")
        print("Logits (last token, first 5):", out.logits[:, -1, :5])

    print("Generating text...")
    
    input_text = "What is the capital of India?"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    
    t2 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50, 
            do_sample=True, 
            temperature=0.7
        )
    print(f"Generation took {time.time() - t2:.2f}s")
        
    print("Output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    run_sarvam_transformers()
