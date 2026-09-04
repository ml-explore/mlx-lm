import ml_collections


def model_config():
    # Qwen3.5-4B, a hybrid of gated delta net and quadratic attention:
    # https://huggingface.co/Qwen/Qwen3.5-4B-Base/blob/main/config.json
    config = ml_collections.ConfigDict()
    config.model_type = "transformer"
    config.hidden_size = 2560
    config.head_dim = 256
    config.vocab_size = 248_320
    config.intermediate_size = 9216
    config.num_attention_heads = 16
    config.num_key_value_heads = 4
    config.num_hidden_layers = 32
    config.rms_norm_eps = 1e-6
    config.rope_theta = 10_000_000
    config.tie_word_embeddings = True
    config.layer_norm = "pre"

    # Every fourth layer is quadratic; the other three are gated delta net.
    config.quadratic_attn_interval = 4
    config.linear_attn_type = "gated_delta"
    config.mlp_type = "mlp"

    # Quadratic layers rotate a quarter of each head and gate their output.
    config.partial_rotary_factor = 0.25
    config.attn_output_gate = True

    # Gated delta net: four value heads per key head.
    config.linear_num_key_heads = 16
    config.linear_num_value_heads = 32
    config.linear_key_head_dim = 128
    config.linear_value_head_dim = 128
    config.linear_conv_kernel_dim = 4
    return config


def optimizer_config():
    config = ml_collections.ConfigDict()
    config.optim = "adam"
    config.weight_decay = 0.1
    config.learning_rate = 1e-4
    config.schedulers = "cosine_decay"
    config.warmup_steps = 1_000
    config.end_learning_rate = 0.0
    config.eps = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.95
    return config


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.model = model_config()
    config.optimizer = optimizer_config()

    config.batch_size = 4  # per GPU
    config.context_size = 4096
    config.grad_accum_steps = 1
    config.num_steps = 1_000_000

    config.max_grad_norm = 5
    config.z_loss_weight = 0
    config.data_type = "bfloat16"
    config.num_valid_batches = 1_000
    # Logging params
    config.steps_per_report = 10
    config.steps_per_checkpoint = 100_000
    config.fsdp_dim = 8
    config.grad_checkpoint = True
    config.tokenizer = "Qwen/Qwen3.5-4B-Base"
    return config
