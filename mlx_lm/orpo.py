
import argparse
import sys
import json
from pathlib import Path

from mlx_lm.tuner.orpo_trainer import run_orpo

def build_parser():
    parser = argparse.ArgumentParser(description="ORPO Training with MLX.")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--adapter-path", type=str, default="adapters.safetensors")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-save", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--lora-parameters", type=json.loads, default='{"rank": 8, "dropout": 0.0, "scale": 20.0}')
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--optimizer-config", type=json.loads, default='{}')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume-adapter-file", type=str, default=None)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()
    run_orpo(args)

if __name__ == "__main__":
    main()
