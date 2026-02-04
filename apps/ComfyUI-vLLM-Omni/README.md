# vLLM-Omni

ComfyUI integration for vLLM-Omni

## Requirement
* Python 3.12
* [ComfyUI installed](https://docs.comfy.org/installation/system_requirements)
* [vLLM-Omni installed](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/installation/) on either the same device or another device discoverable via the internet.
* No need to install additional packages apart from those already required by ComfyUI.

> [!TIP]
> If you run both ComfyUI and vLLM-Omni on the same device, you can create separate virtual environments and use different Python versions for them.


## Installation

Copy this folder to the `custom_nodes` subfolder of your ComfyUI installation. Your directory should look like `ComfyUI/custom_nodes/ComfyUI-vLLM-Omni`.

If you are running ComfyUI during copying, you should restart ComfyUI to load this extension.

> [!TIP]
> You can use utility websites such as https://download-directory.github.io/ to download a subdirectory of a repo. Also checkout community discussions (e.g., https://stackoverflow.com/questions/7106012/download-a-single-folder-or-directory-from-a-github-repository) for more info.

On the device and virtual environment you run ComfyUI, launch ComfyUI with
```bash
cd ComfyUI

# The regular way
python main.py

# If you are mainly using this node, launch it faster with
python main --cpu
```

On the device and virtual environment you run vLLM-Omni, start a model service with
```bash
vllm serve The_Model_ID_to_Serve --omni --port 8000
```

Check **ComfyUI's sidebar -> Node Library**. There should be a new folder named **vLLM-Omni**.
If no, check your shell running the ComfyUI process. There may be some error messages before the line `Import times for custom nodes:` and the line `To see the GUI go to: http://127.0.0.1:8188`.

## Quickstart

This extension offers the following nodes based on the output modalities:
* **Generate Image** for text-to-image and image-to-image tasks
* **Multimodality Comprehension** for multimodality-to-text and multimodality-to-audio tasks
* **Generate Audio** for text-to-audio tasks
* **Generate Video** ~~for text-to-video and image-to-video tasks~~ (Not yet implemented)

> [!INFO]
> The node UI and feature designs are intended to match vLLM-Omni online serving interfaces. It cannot offer more than what the interfaces support.

To build a simple workflow,
* Drag a generation node onto the canvas.
* Depending on your need, grab built-in multimedia file loader nodes, such as **image->Load Image**, **image->video->Load Video**, **audio->Load Audio**
* Depending on your need, grab built-in multimedia file preview nodes, such as **image->Preview Image**, **image->video->Save Video**, **audio->Preview Audio**. For text output, you can install [ComfyUI-Custom-Scripts plugin](https://github.com/pythongosssss/ComfyUI-Custom-Scripts/) and grab its **utils->Show Text 🐍** node.
* If you want to tune sampling parameters, grab corresponding nodes from **vLLM-Omni-> Sampling Params**.
    * For multi-stage models, you can connect multiple **AR Sampling Params** and **Diffusion Sampling Params** nodes to a **Multi-Stage Sampling Params List** node, and connect this node to the generation node.
    * For some multi-stage models like BAGEL, [only one stage's sampling parameters are exposed and tunable via vLLM-Omni's online serving API](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/online_serving/bagel/). Thus, these models are treated as single-stage ones. Please check the vLLM-Omni documentation on how to use correctly set each model's sampling parameters.
    * For multi-stage models where all stages are either autoregression or diffusion, you can also connect only a single Sampling Params node, indicating that this set of sampling parameters will be used for all stages.


## Develop

Follow the [development convention and rules of vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/contributing/).

## Limitation and Non-Goals

* Single server mode only. No automatic load balancing or failover.
* Features set is bounded to vLLM-Omni's online service capability, including
    * The types of models supported in online mode,
    * The types of sampling parameters supported in the online mode,
    * The ways to send files (primarily through full-length base64 in JSON payload),
    * (The lack of) Authentication
    * (The lack of) Progress indicator

## Support

If you are new to ComfyUI, please check out [its documentation](https://docs.comfy.org/) for usage instructions.

If you are new to vLLM-Omni, please also check out [its documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/) for usage instructions.

Whenever you find an issue or problem, please
* First find out if this is an upstream limitation of vLLM-Omni's online serving mode, by [checking their documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/examples/).
* [Open an issue](https://github.com/vllm-project/vllm-omni/issues) that clearly describes this ComfyUI or online service problem.

## Acknowledgements

Features
* https://github.com/dougbtv/comfyui-vllm-omni/ The official reference implementation for ComfyUI integration with vLLM-Omni's DALL-E compatible image generation API.
* https://github.com/Comfy-Org/ComfyUI/tree/master/comfy_extras ComfyUI's built-in node implementations.

UI/UX design references
* https://github.com/sgl-project/sglang/pull/15271 SGLang Diffusion's official ComfyUI integration for image and video generation.
* https://github.com/SXQBW/ComfyUI-Qwen-Omni A third party ComfyUI integration for Qwen Omni series.
* https://github.com/flybirdxx/ComfyUI-Qwen-TTS https://github.com/DarioFT/ComfyUI-Qwen3-TTS Tthird party ComfyUI integrations for Qwen TTS series.
