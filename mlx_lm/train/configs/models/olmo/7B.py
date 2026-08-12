import ml_collections


def bolt_config():
    config = ml_collections.ConfigDict()
    config.name = "olmo_7B"
    config.gpu_type = "b200"
    config.num_gpus = 16
    config.tags = ["16gpu", "olmo_7B"]
    return config


def model_config():
    # https://huggingface.co/allenai/Olmo-3-1025-7B/blob/main/config.json
    config = ml_collections.ConfigDict()
    config.model_type = "transformer"
    config.hidden_size = 4096
    config.head_dim = 128
    config.vocab_size = 128_256
    config.intermediate_size = 11008
    config.num_attention_heads = 32
    config.num_key_value_heads = 32
    config.num_hidden_layers = 32
    config.rms_norm_eps = 1e-6
    config.rope_theta = 500_000
    config.tie_word_embeddings = False
    config.layer_norm = "post"

    return config


def optimizer_config():
    # this configuretion is based on olmo paper
    # to make it valid, context length should be 8192
    # and global batch size should be 512
    # (e.g. batch size 1 per GPU, grad accumulation steps 4, 128 GPUs)
    # therefore decay_steps+warmup_steps is unchanged and equal to 1_413_824 which
    # results in ~5.93T tokens seens during training
    config = ml_collections.ConfigDict()
    config.optim = "adamw"
    config.weight_decay = 0.1
    config.learning_rate = 3e-4
    config.schedulers = "cosine_decay"
    config.warmup_steps = 2_000
    config.end_learning_rate = 3e-5
    config.decay_steps = 1_413_824 - 2_000
    config.eps = 1e-8
    config.beta1 = 0.9
    config.beta2 = 0.95

    return config


def get_config():
    config = ml_collections.ConfigDict()
    config.seed = 0
    config.model = model_config()
    config.optimizer = optimizer_config()
    config.bolt = bolt_config()

    # Optimization params
    config.batch_size = 2  # per GPU
    config.context_size = 8192
    config.grad_accum_steps = 1
    config.num_steps = 2_827_648  # 1_406_656 * 8192 * 2 * 256 ~ 5.93T tokens

    config.max_grad_norm = 1
    config.z_loss_weight = 1e-5
    config.data_type = "bfloat16"
    config.num_valid_batches = 1_000
    # Logging params
    config.steps_per_report = 10
    config.steps_per_checkpoint = 100_000
    config.fsdp_dim = 8
    config.reduction_size = 32 * 1024 * 1024  # 32MB
    config.grad_checkpoint = False

    return config
