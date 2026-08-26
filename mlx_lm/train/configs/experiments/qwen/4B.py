import ml_collections


def model_config():
    config = ml_collections.ConfigDict()
    config.model_type = "transformer"
    config.hidden_size = 2560
    config.head_dim = 128
    config.vocab_size = 128_256
    config.intermediate_size = 9728
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.num_hidden_layers = 36
    config.tie_word_embeddings = True
    config.layer_norm = "pre"
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

    # Optimization params
    config.batch_size = 2  # per GPU
    config.context_size = 8192
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
    config.grad_checkpoint = False

    return config
