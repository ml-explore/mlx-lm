import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.source = "hf"
    config.name = "allenai/dolma3_mix-6T"
    config.text_key = "text"
    config.shuffle_buffer = 0
    return config
