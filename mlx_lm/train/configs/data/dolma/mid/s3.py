import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.source = "s3"
    config.uri = "s3://smollm/dolma3_dolmino_mix-100B-1025-shuffled/data/"
    config.num_groups = 64
    config.shards_per_group = 32
    config.suffix = ".json.gz"
    config.text_key = "text"
    config.shuffle_buffer = 0
    return config
