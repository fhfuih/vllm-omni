MODEL_PIPELINE_SPECS = {
    "default-image-model": {
        "modalities": ["image"],
        "stages": ["autoregressive", "diffusion"],
        "required_inputs": ["image", "audio", "video"],
    },
    "default-video-model": {
        "modalities": ["video"],
        "stages": ["autoregressive", "diffusion"],
        "required_inputs": ["image", "audio"],
    },
    "default-audio-model": {
        "modalities": ["audio"],
        "stages": ["autoregressive"],
        "required_inputs": ["image", "audio", "video"],
    },
    "ByteDance-Seed/BAGEL-7B-MoT": {
        "input_modalities": ["image"],
        "stages": ["autoregressive", "diffusion"],
        "extra_fields": {
            "modalities": ["text", "image"],
        },
    },
    "Qwen/Qwen2.5-Omni-7B": {
        "input_modalities": ["image", "audio", "video"],
        "stages": ["autoregressive", "autoregressive", "autoregressive"],
        "extra_fields": {
            "modalities": ["text", "audio"],
        },
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "input_modalities": ["image", "audio", "video"],
        "stages": ["autoregressive", "autoregressive", "autoregressive"],
        "extra_fields": {
            "modalities": ["text", "audio"],
        },
    },
}
