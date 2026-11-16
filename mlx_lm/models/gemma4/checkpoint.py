# Copyright © 2025 Apple Inc.


__all__ = [
    "sanitize",
]


def sanitize(model, weights):
    new_weights = {}
    for k, v in weights.items():
        starts_w_model = k.startswith("model.")

        k = k.removeprefix("model.")
        if k.startswith(
            (
                "vision_tower",
                "multi_modal_projector",
                "audio_tower",
                "embed_audio",
                "embed_vision",
                "vision_embedder",  # gemma4_unified encoder-free vision
            )
        ):
            continue

        if not starts_w_model:
            new_weights[k] = v
            continue

        if k.startswith("language_model"):
            k = k.replace("language_model.", "language_model.model.")

        new_weights[k] = v

    return model.language_model.sanitize(new_weights)
