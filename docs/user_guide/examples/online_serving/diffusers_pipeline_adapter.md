# Diffusers Backend Adapter

Source <https://github.com/vllm-project/vllm-omni/tree/main/examples/online_serving/diffusers_pipeline_adapter>.


This example demonstrates how to serve any 🤗 Diffusers pipeline through vLLM-Omni
using the `diffusers` load format.

## Supported Models

Any model loadable via `DiffusionPipeline.from_pretrained()` should be supported, including text-to-image, image-to-image, text-to-video, image-to-video, and text-to-audio.

## Usage

```bash
vllm serve "stable-diffusion-v1-5/stable-diffusion-v1-5" \
    --omni \
    --diffusion-load-format diffusers
```

Users turns on diffusers backend primarily through `--diffusion-load-format diffusers` argument.
There are two more optional arguments, `--diffusers-load-kwargs` and `--diffusers-call-kwargs`,
which are are only valid together with `--diffusion-load-format diffusers`.

After launching the model, users send a request as usual. Refer to other documentation pages on how to request a particular input/output modality, such as `examples/online_serving/text_to_image/openai_chat_client.py`.

## Configuration Reference

### `--diffusers-load-kwargs`

Passed as-is to `DiffusionPipeline.from_pretrained()`.

This is suitable for model-specific configurations not available through the vLLM-Omni interface.
For example: `--diffusers-load-kwargs '{"use_safetensors": true}'`.

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_load_kwargs`, the **latter** will take precedence.

### `--diffusers-call-kwargs`

Passed to `pipeline.__call__()`.

This is suitable for sampling parameters not available through the vLLM-Omni interface (such as online serving payloads).

When a parameter is available in the vLLM-Omni interface, it will be adapted here.
But if that parameter is simultaneously set in both the vLLM-Omni interface and `diffusers_call_kwargs`, the **former** will take precedence (because it is set at request time).
