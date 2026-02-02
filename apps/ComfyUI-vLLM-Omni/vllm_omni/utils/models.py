def _qwen25_payload_preprocessor(payload: dict) -> dict:
    if payload["messages"][0]["role"] != "system":
        payload["messages"] = [
            {
                "role": "system",
                "content": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.",
            },
            *payload["messages"],
        ]
    return payload


MODEL_PIPELINE_SPECS = {
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
        "payload_preprocessor": _qwen25_payload_preprocessor,
    },
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": {
        "input_modalities": ["image", "audio", "video"],
        "stages": ["autoregressive", "autoregressive", "autoregressive"],
        "extra_fields": {
            "modalities": ["text", "audio"],
        },
    },
}
