import re
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Pattern,
    TypeAlias,
    TypedDict,
)

PayloadPreprocessor: TypeAlias = Callable[[dict[str, Any]], dict[str, Any]]


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


class Spec(TypedDict):
    input_modalities: list[str]
    stages: list[str]
    extra_fields: Dict[str, Any]
    payload_preprocessor: Optional[PayloadPreprocessor]


MODEL_PIPELINE_SPECS = {
    r"BAGEL-7B-MoT": {
        "input_modalities": ["image"],
        "stages": ["autoregression", "diffusion"],
        "extra_fields": {
            "modalities": ["text", "image"],
        },
    },
    r"Qwen2.5-Omni*": {
        "input_modalities": ["image", "audio", "video"],
        "stages": ["autoregression", "autoregression", "autoregression"],
        "extra_fields": {
            "modalities": ["text", "audio"],
        },
        "payload_preprocessor": _qwen25_payload_preprocessor,
    },
    r"Qwen3-Omni*": {
        "input_modalities": ["image", "audio", "video"],
        "stages": ["autoregression", "autoregression", "autoregression"],
        "extra_fields": {
            "modalities": ["text", "audio"],
        },
    },
}
_MODEL_PIPELINE_SPECS: dict[Pattern, Spec] = {}
for k, v in MODEL_PIPELINE_SPECS.items():
    _MODEL_PIPELINE_SPECS[re.compile(k)] = v  # type: ignore
del MODEL_PIPELINE_SPECS
MODEL_PIPELINE_SPECS = _MODEL_PIPELINE_SPECS


def lookup_model_spec(model: str) -> Optional[Spec]:
    last_component = model.rstrip("/").rsplit("/", 1)[1]
    for pattern, spec in MODEL_PIPELINE_SPECS.items():
        if pattern.search(last_component):
            return spec


# ============== DEMONSTRATION ==============

if __name__ == "__main__":
    test_paths = [
        "Qwen/Qwen2.5-Omni-7B",
        "MyModels/Qwen2.5-Omni-3B",
        "/root/home/Qwen2.5-Omni-7B",
        "Qwen/Qwen3-Omni",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "Custom/Path/UnknownModel-Instruct",
        "Not/Matching/Anything",
    ]

    test_payload = {"messages": [{"role": "user", "content": "prompt"}]}

    print("Testing registry lookups:\n")
    for path in test_paths:
        spec = lookup_model_spec(path)
        if spec:
            if preprocessor := spec.get("payload_preprocessor"):
                result = preprocessor(test_payload)
                print(f"✓ {path:<40} → {result}")
            else:
                print(f"✓ {path:<40} → No preprocessor")
        else:
            print(f"✗ {path:<40} → No match")
