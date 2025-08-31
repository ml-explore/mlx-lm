# MLX-LM DPO Implementation Guide

## Overview
This document provides a comprehensive guide for implementing Direct Preference Optimization (DPO) in the mlx-lm repository. DPO is a method for training language models to align with human preferences without requiring a separate reward model.

## Repository Structure Understanding

### Key Files to Modify/Create:
```
mlx_lm/
├── tuner/
│   ├── losses.py          # Add DPO loss function
│   ├── datasets.py        # Add PreferenceDataset class
│   ├── trainer.py         # Modify for dual forward passes
│   └── dpo.py             # New: Main DPO training logic
├── __main__.py            # Add DPO subcommand
└── dpo_train.py           # New: CLI entry point
```

## Implementation Plan

### Phase 1: Core DPO Loss Function

**File: `mlx_lm/tuner/losses.py`**

Add the DPO loss implementation following the existing pattern of KL divergence kernels:

```python
def dpo_loss(
    policy_chosen_logits: mx.array,
    policy_rejected_logits: mx.array, 
    reference_chosen_logits: mx.array,
    reference_rejected_logits: mx.array,
    beta: float = 0.1
) -> mx.array:
    """
    Direct Preference Optimization loss function.
    
    Args:
        policy_chosen_logits: Logits from policy model for chosen responses
        policy_rejected_logits: Logits from policy model for rejected responses  
        reference_chosen_logits: Logits from reference model for chosen responses
        reference_rejected_logits: Logits from reference model for rejected responses
        beta: Temperature parameter for DPO
        
    Returns:
        DPO loss value
    """
    # Implementation details:
    # 1. Compute log probabilities for chosen and rejected sequences
    # 2. Calculate KL-regularized reward differences
    # 3. Apply Bradley-Terry preference model
    # 4. Consider Metal kernel optimization for performance
```

**Mathematical Formula:**
```
L_DPO = -E[(x,y_w,y_l)~D][log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]
```

### Phase 2: Preference Dataset Handler

**File: `mlx_lm/tuner/datasets.py`**

Create a new dataset class to handle preference data:

```python
class PreferenceDataset:
    """
    Dataset for DPO training with chosen/rejected pairs.
    
    Expected format:
    {
        "prompt": "...",
        "chosen": "...", 
        "rejected": "..."
    }
    """
    
    def __init__(
        self,
        data: List[Dict[str, str]], 
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048
    ):
        # Implementation details:
        # 1. Process prompt + chosen/rejected pairs
        # 2. Tokenize all sequences
        # 3. Handle padding and truncation
        # 4. Create attention masks for loss computation
        
    def __getitem__(self, idx: int):
        # Return: (prompt_tokens, chosen_tokens, rejected_tokens, lengths)
        pass
```

### Phase 3: DPO Trainer Integration

**File: `mlx_lm/tuner/trainer.py`**

Modify the training loop to support DPO:

```python
def dpo_step(
    policy_model,
    reference_model, 
    batch,
    optimizer,
    beta: float = 0.1
):
    """
    Single DPO training step with dual forward passes.
    
    Args:
        policy_model: Model being trained
        reference_model: Frozen reference model
        batch: Batch of preference data
        optimizer: Optimizer instance
        beta: DPO temperature parameter
    """
    # Implementation:
    # 1. Forward pass through policy model (chosen + rejected)
    # 2. Forward pass through reference model (chosen + rejected) 
    # 3. Compute DPO loss
    # 4. Backward pass and optimizer step
    # 5. Return loss and metrics
```

**Key considerations:**
- Reference model should be frozen and not updated
- Batch processing for efficiency
- Memory management for dual forward passes
- Gradient accumulation support

### Phase 4: Main DPO Training Script

**File: `mlx_lm/dpo.py`**

Create the main DPO training logic following `lora.py` exactly:

```python
def train_model(
    args,
    model: nn.Module,
    train_set,
    valid_set,
    training_callback: TrainingCallback = None,
):
    """
    DPO training with same model setup logic as lora.py
    """
    mx.random.seed(args.seed)
    model.freeze()
    
    # EXACT SAME LOGIC AS lora.py
    if args.fine_tune_type == "full":
        for l in model.layers[-max(args.num_layers, 0):]:
            l.unfreeze()
        args.lora_parameters = None
    elif args.fine_tune_type in ["lora", "dora"]:
        linear_to_lora_layers(
            model,
            args.num_layers,
            args.lora_parameters,
            use_dora=(args.fine_tune_type == "dora"),
        )
    else:
        raise ValueError(f"Received unknown fine-tune-type {args.fine_tune_type}")

    # Load reference model (frozen)
    reference_model = load_reference_model(args)
    
    # Use same TrainingArgs, add DPO-specific params to args directly
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        # ... same as lora.py
    )
    
    # Custom DPO training loop with dual forward passes
    train_dpo(model, reference_model, training_args, ...)
```

### Phase 5: CLI Integration

**File: `mlx_lm/dpo_train.py`**

Create CLI entry point following the existing LoRA pattern:

```python
def build_parser():
    """Build argument parser for DPO training."""
    parser = argparse.ArgumentParser(description="DPO finetuning.")
    
    # Inherit base arguments from LoRA
    # Add DPO-specific arguments:
    parser.add_argument("--beta", type=float, default=0.1, 
                       help="DPO temperature parameter")
    parser.add_argument("--reference-model", type=str,
                       help="Path to reference model (defaults to initial policy model)")
    # ... other DPO arguments

def main():
    """Main entry point following mlx-lm patterns."""
    # Parse arguments, load models, train
    pass
```

**File: `mlx_lm/__main__.py`**

Add DPO to subcommands:
```python
subcommands = {
    # ... existing commands
    "dpo",  # Add this line
}
```

## Data Format Requirements

### Input Data Format
```json
{
    "prompt": "Explain the concept of machine learning",
    "chosen": "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed...",
    "rejected": "Machine learning is just computers doing stuff automatically..."
}
```

### Alternative Formats to Support
```json
// Chat format with preferences  
{
    "messages": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "chosen_response"}
    ],
    "rejected": [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "rejected_response"}  
    ]
}
```

## Technical Implementation Details

### Memory Optimization Strategies
1. **Gradient Checkpointing**: Use existing `grad_checkpoint` functionality
2. **Sequence Packing**: Pack multiple short sequences into single batch
3. **Reference Model Caching**: Cache reference model outputs when possible
4. **Mixed Precision**: Leverage MLX's automatic mixed precision

### Performance Considerations
1. **Metal Kernels**: Consider custom Metal shaders for DPO loss computation
2. **Batch Size**: Balance between memory usage and training stability
3. **Sequence Length**: Implement dynamic padding to reduce computation waste

### Evaluation Metrics
Add DPO-specific metrics to the callback system:
- Preference accuracy (chosen vs rejected win rate)
- KL divergence from reference model
- Average reward margins
- Policy vs reference model perplexity comparison

## Testing Strategy

### Unit Tests
1. **Loss Function**: Test DPO loss computation with known inputs
2. **Dataset**: Verify preference pair processing and batching
3. **Training Step**: Validate single training step mechanics

### Integration Tests  
1. **End-to-End Training**: Small dataset, few iterations
2. **Checkpoint Loading**: Verify model saving/loading
3. **Distributed Training**: Multi-GPU functionality

### Validation Datasets
Use public preference datasets for validation:
- Anthropic HH-RLHF dataset
- Stanford SHP dataset  
- OpenAI WebGPT comparisons

## Configuration Examples

### Basic DPO Training Config (YAML)
```yaml
# dpo_config.yaml
model: "microsoft/DialoGPT-small"
train: true
fine_tune_type: "lora"  # lora/dora/full - same as SFT
data: "path/to/preference/data"
beta: 0.1
batch_size: 4
iters: 1000
learning_rate: 1e-6
max_seq_length: 1024
lora_parameters:
  rank: 8
  dropout: 0.0
  scale: 20.0
reference_model: null  # Uses initial policy model
```

### Advanced Configuration  
```yaml
model: "meta-llama/Llama-2-7b-hf"
train: true
fine_tune_type: "dora"  # lora/dora/full - same as SFT
data: "anthropic/hh-rlhf"
beta: 0.2
batch_size: 8
iters: 5000
learning_rate: 5e-7
max_seq_length: 2048
lora_parameters:
  rank: 16
  dropout: 0.1
  scale: 10.0
reference_model: "path/to/reference/model"
grad_checkpoint: true
optimizer: "adamw"
lr_schedule:
  name: "cosine"
  warmup_steps: 100
report_to: "wandb"
project_name: "dpo-llama-experiments"
```

## Integration Points with Existing Code

### Reuse Existing Infrastructure
1. **Trainer Loop**: Extend `mlx_lm/tuner/trainer.py:train()`
2. **Dataset Loading**: Use existing `load_dataset()` patterns  
3. **Model Management**: Leverage `mlx_lm/utils.py:load()` and `save_config()`
4. **Optimization**: Use existing optimizer and scheduler setup
5. **Callbacks**: Integrate with existing reporting systems

### Metal Kernel Optimization
Follow the pattern from `kl_div_loss` and `js_div_loss`:
1. Create custom Metal shaders for DPO loss computation
2. Implement both forward and backward passes  
3. Add CPU fallback for non-Metal environments
4. Use `@mx.custom_function` decorator pattern

## Expected Performance Characteristics

### Training Speed
- **Single GPU**: Expect ~2-4x slower than standard LoRA due to dual forward passes
- **Memory Usage**: ~1.5-2x base model memory (policy + reference models)
- **Convergence**: Typically 1000-5000 iterations depending on dataset size

### Quality Expectations
- **Win Rate**: Target >60% preference accuracy on held-out data
- **KL Divergence**: Monitor drift from reference model (target < 2.0)
- **Perplexity**: Should maintain or improve on evaluation data

## Error Handling and Edge Cases

### Common Issues to Handle
1. **Sequence Length Mismatch**: Different lengths for chosen/rejected pairs
2. **Memory Overflow**: Large batch sizes with long sequences  
3. **Numerical Instability**: Log probabilities and sigmoid computations
4. **Reference Model Mismatch**: Different tokenizers or architectures

### Debugging Tools
1. **Loss Debugging**: Log individual components (chosen/rejected rewards)
2. **Gradient Analysis**: Monitor gradient norms and distributions
3. **Preference Validation**: Sample and manually inspect preference decisions

## Future Extensions

### Potential Enhancements
1. **ORPO Support**: Odds Ratio Preference Optimization
2. **CPO Integration**: Constrained Policy Optimization  
3. **Multi-Objective DPO**: Handle multiple preference criteria
4. **Online DPO**: Real-time preference collection and training

### Research Directions
1. **Advanced Regularization**: Beyond KL divergence  
2. **Hierarchical Preferences**: Handle preference hierarchies
3. **Uncertainty Quantification**: Model preference confidence
4. **Efficient Reference Models**: Distillation and compression techniques

## MLX-LM Contributing Guidelines Compliance

Based on the official CONTRIBUTING.md, our DPO implementation must follow these standards:

### Code Quality Requirements
1. **Testing**: Every new function must have corresponding tests
2. **Code Formatting**: Use `black` for Python and `clang-format` for C++
3. **Pre-commit Hooks**: Install and use pre-commit for automatic formatting
4. **Reviews**: All PRs require at least one review and passing tests

### Required Setup Commands
```bash
# Install development dependencies
pip install -e .
pip install pre-commit

# Setup pre-commit hooks
pre-commit install

# Run formatting manually if needed
black mlx_lm/tuner/dpo.py
pre-commit run --files mlx_lm/tuner/losses.py

# Run all tests
python -m unittest discover tests/
```

### Testing Requirements
Create comprehensive test files following mlx-lm patterns:

**File: `tests/test_dpo.py`**
```python
import unittest
import mlx.core as mx
import numpy as np
from mlx_lm.tuner.losses import dpo_loss
from mlx_lm.tuner.datasets import PreferenceDataset

class TestDPO(unittest.TestCase):
    def test_dpo_loss_computation(self):
        # Test DPO loss with known inputs/outputs
        pass
        
    def test_preference_dataset_loading(self):
        # Test dataset processing for preference pairs
        pass
        
    def test_dpo_training_step(self):
        # Test single training iteration
        pass

if __name__ == "__main__":
    unittest.main()
```

### File Structure Compliance
Following mlx-lm conventions for new model features:

```
mlx_lm/
├── tuner/
│   ├── losses.py          # Add dpo_loss function
│   ├── datasets.py        # Add PreferenceDataset class  
│   ├── trainer.py         # Modify existing functions
│   └── utils.py           # Add DPO-specific utilities
├── dpo.py                 # Main DPO entry point (like lora.py)
├── __main__.py            # Add "dpo" to subcommands
└── tests/
    └── test_dpo.py        # Comprehensive test suite
```

### Documentation Standards
1. **Docstrings**: Follow existing patterns with clear parameter descriptions
2. **Type Hints**: Use proper type annotations for all functions
3. **Examples**: Include usage examples in docstrings

**Example following mlx-lm style:**
```python
def dpo_loss(
    policy_chosen_logits: mx.array,
    policy_rejected_logits: mx.array,
    reference_chosen_logits: mx.array, 
    reference_rejected_logits: mx.array,
    beta: float = 0.1,
) -> mx.array:
    """
    Compute Direct Preference Optimization loss.
    
    Args:
        policy_chosen_logits: Policy model logits for chosen responses [batch, seq, vocab]
        policy_rejected_logits: Policy model logits for rejected responses [batch, seq, vocab]
        reference_chosen_logits: Reference model logits for chosen responses [batch, seq, vocab]
        reference_rejected_logits: Reference model logits for rejected responses [batch, seq, vocab]
        beta: Temperature parameter controlling strength of KL penalty
        
    Returns:
        DPO loss scalar value
        
    Examples:
        >>> policy_logits_chosen = mx.random.normal((2, 10, 1000))
        >>> policy_logits_rejected = mx.random.normal((2, 10, 1000))
        >>> ref_logits_chosen = mx.random.normal((2, 10, 1000))  
        >>> ref_logits_rejected = mx.random.normal((2, 10, 1000))
        >>> loss = dpo_loss(policy_logits_chosen, policy_logits_rejected, 
        ...                 ref_logits_chosen, ref_logits_rejected, beta=0.1)
    """
```

## Implementation Checklist

### Phase 1: Core Implementation ✓
- [ ] DPO loss function in `losses.py` with comprehensive docstrings
- [ ] PreferenceDataset class in `datasets.py` with type hints
- [ ] Unit tests in `tests/test_dpo.py` for loss computation
- [ ] Code formatted with `black` and pre-commit hooks passing

### Phase 2: Integration ✓  
- [ ] CLI interface in `dpo.py` following `lora.py` patterns
- [ ] Configuration system integration with proper YAML support
- [ ] Update `__main__.py` to include "dpo" subcommand
- [ ] Integration tests for end-to-end functionality

### Phase 3: Optimization ✓
- [ ] Metal kernel optimization for DPO loss (if performance critical)
- [ ] Memory efficiency improvements and gradient checkpointing support
- [ ] Distributed training compatibility testing
- [ ] Performance benchmarking against reference implementations

### Phase 4: Contribution Readiness ✓
- [ ] All tests passing: `python -m unittest discover tests/`
- [ ] Code formatting verified: `pre-commit run --all-files`
- [ ] Documentation complete with examples and type hints
- [ ] PR ready with clear description and test coverage

### Pre-Submission Checklist ✓
- [ ] Fork mlx-lm repository
- [ ] Create feature branch: `git checkout -b feature/dpo-implementation`
- [ ] Install development dependencies: `pip install -e .`
- [ ] Setup pre-commit: `pre-commit install`
- [ ] Run all formatters: `pre-commit run --all-files`
- [ ] Run all tests: `python -m unittest discover tests/`
- [ ] Verify no regressions in existing functionality
- [ ] Write clear commit messages and PR description

### Quality Assurance Standards
1. **Test Coverage**: Aim for >90% test coverage for new code
2. **Performance**: Document any performance implications
3. **Memory Usage**: Profile memory usage for large models
4. **Backwards Compatibility**: Ensure no breaking changes to existing APIs
5. **Error Handling**: Robust error messages and edge case handling

This implementation guide provides the foundation for building a robust, production-ready DPO implementation that integrates seamlessly with the existing mlx-lm codebase while following established patterns and best practices **and meeting all MLX-LM contributing standards**.