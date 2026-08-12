import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.source = "hf"
    config.name = "allenai/dolma3_dolmino_mix-100B-1025"
    config.text_key = "text"
    config.shuffle_buffer = 0
    return config
