from transformers.configuration_utils import PretrainedConfig


class SarvamMoEConfig(PretrainedConfig):
    model_type = "sarvam_moe"
    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=8192,
        num_hidden_layers=19,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="silu",
        use_qkv_bias=False,
        use_bias=False,
        rms_norm_eps=1e-06,
        tie_word_embeddings=False,
        embedding_dropout=0.0,
        attention_dropout=0.0,
        output_dropout=0.0,
        initializer_range=0.006,
        max_position_embeddings=4096,
        rope_theta=10000.0,
        use_cache=True,
        max_window_layers=19,
        rope_scaling=None,
        pad_token_id=0,
        eos_token_id=1,
        num_experts=128,
        num_shared_experts=1,
        num_experts_per_tok=6,
        n_group=1,
        topk_group=1,
        moe_intermediate_size=1024,
        first_k_dense_replace=1,
        head_dim=256,
        output_router_logits=False,
        use_qk_norm=True,
        moe_router_enable_expert_bias=True,
        routed_scaling_factor=2.5,
        attn_implementation: str = "eager",
        **kwargs,
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.rope_scaling = rope_scaling
        self.use_qk_norm = use_qk_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.attn_implementation = attn_implementation
        self._attn_implementation = attn_implementation

        self.base_model_tp_plan = {
            "layers.*.attention.query_key_value": "colwise",
            "layers.*.attention.dense": "rowwise",
            "layers.*.mlp.gate_proj": "colwise",
            "layers.*.mlp.up_proj": "colwise",
            "layers.*.mlp.down_proj": "rowwise",
            "layers.*.mlp.experts.*.gate_proj": "colwise",
            "layers.*.mlp.experts.*.up_proj": "colwise",
            "layers.*.mlp.experts.*.down_proj": "rowwise",
            "layers.*.mlp.shared_experts.gate_proj": "colwise",
            "layers.*.mlp.shared_experts.up_proj": "colwise",
            "layers.*.mlp.shared_experts.down_proj": "rowwise",
        }
        self.base_model_pp_plan = {
            "word_embeddings": (["input_ids"], ["inputs_embeds"]),
            "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
            "norm": (["hidden_states"], ["hidden_states"]),
        }

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
