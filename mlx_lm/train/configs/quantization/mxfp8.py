import ml_collections


def get_config():

    config = ml_collections.ConfigDict()
    config.qkv_dtype = "mxfp8"
    config.mlp_dtype = "mxfp8"
    config.embedding_dtype = "mxfp8"
    config.lm_head_dtype = "mxfp8"

    return config
