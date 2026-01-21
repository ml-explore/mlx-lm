
import argparse
import sys
from pathlib import Path

def main():
    try:
        from lm_eval import evaluator
        from lm_eval.models.huggingface import HFLM
        from .internal_eval import MLXLM
    except ImportError:
        print("Error: lm-evaluation-harness is not installed.")
        print("Please install it with: pip install lm-eval")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Evaluate MLX models with lm-evaluation-harness.")
    parser.add_argument("--model", type=str, required=True, help="Path to model or HF repo")
    parser.add_argument("--tasks", type=str, default="mmlu", help="Comma separated list of tasks")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples per task")
    parser.add_argument("--output-path", type=str, default=None, help="Output path for results")
    parser.add_argument("--adapter-path", type=str, default=None, help="Adapter path")
    
    args = parser.parse_args()

    lm = MLXLM(
        model=args.model,
        batch_size=args.batch_size,
        adapter_path=args.adapter_path
    )
    
    tasks = args.tasks.split(",")
    
    print(f"Evaluating {args.model} on {tasks}")
    
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )
    
    # Print table
    from lm_eval.utils import make_table
    print(make_table(results))
    
    if args.output_path:
        import json
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
