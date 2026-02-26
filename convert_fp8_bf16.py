
import os
import glob
import torch
import shutil
import json
import argparse
from safetensors.torch import load_file, save_file
from tqdm import tqdm

def convert_fp8_to_bf16(src_dir, dst_dir):
    print(f"Converting FP8 weights from {src_dir} to BF16 in {dst_dir}...")
    
    if os.path.exists(dst_dir):
        print(f"Destination {dst_dir} exists, skipping creation.")
    else:
        os.makedirs(dst_dir)

    # Copy config files
    for filename in os.listdir(src_dir):
        if filename.endswith(".json") or filename.endswith(".py") or filename.endswith(".model") or filename.endswith(".txt"):
            src_file = os.path.join(src_dir, filename)
            dst_file = os.path.join(dst_dir, filename)
            shutil.copy2(src_file, dst_file)
            print(f"Copied {filename}")

    # Process safetensors
    safetensors_files = glob.glob(os.path.join(src_dir, "*.safetensors"))
    if not safetensors_files:
        # Check for recursive files or check if they are downloading
        safetensors_files = glob.glob(os.path.join(src_dir, "**/*.safetensors"), recursive=True)
    
    if not safetensors_files:
        print("No safetensors found! Conversion aborted.")
        return

    for st_file in tqdm(safetensors_files, desc="Converting files"):
        print(f"Processing {st_file}...")
        try:
            state_dict = load_file(st_file)
            new_state_dict = {}
            for k, v in state_dict.items():
                if v.dtype == torch.float8_e4m3fn:
                    # Convert to BF16
                    new_state_dict[k] = v.to(torch.bfloat16)
                elif v.dtype == torch.float8_e5m2:
                    new_state_dict[k] = v.to(torch.bfloat16)
                else:
                    new_state_dict[k] = v # Keep as is (usually already bf16 or float32 for norms)
            
            # Save to destination
            basename = os.path.basename(st_file)
            save_path = os.path.join(dst_dir, basename)
            save_file(new_state_dict, save_path)
            print(f"Saved {save_path}")
            
            # Clear memory
            del state_dict
            del new_state_dict
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error converting {st_file}: {e}")

    # Update config.json to remove quantization info if present, so mlx loads it as standard model
    config_path = os.path.join(dst_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Remove quantization config which might trigger mlx logic
        if "quantization_config" in config:
            del config["quantization_config"]
            print("Removed quantization_config from config.json")
            
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Source directory with FP8 weights")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory for BF16 weights")
    args = parser.parse_args()
    
    convert_fp8_to_bf16(args.src, args.dst)
