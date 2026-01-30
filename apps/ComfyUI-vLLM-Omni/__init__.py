"""Top-level package for vllm_omni."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Zeyu Huang"""
__email__ = "11222265+fhfuih@users.noreply.github.com"
__version__ = "0.0.1"

from .vllm_omni.nodes import (
    VLLMOmniGenerateImage,
    VLLMOmniGenerateVideo,
    VLLMOmniGenerateAudio,
    VLLMOmniARSampling,
    VLLMOmniDiffusionSampling,
    VLLMOmniSamplingParamsList,
)

# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "VLLMOmniGenerateImage": VLLMOmniGenerateImage,
    "VLLMOmniGenerateVideo": VLLMOmniGenerateVideo,
    "VLLMOmniGenerateAudio": VLLMOmniGenerateAudio,
    "VLLMOmniARSampling": VLLMOmniARSampling,
    "VLLMOmniDiffusionSampling": VLLMOmniDiffusionSampling,
    "VLLMOmniSamplingParamsList": VLLMOmniSamplingParamsList,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VLLMOmniGenerateImage": "Generate Image",
    "VLLMOmniGenerateVideo": "Generate Video",
    "VLLMOmniGenerateAudio": "Generate Audio",
    "VLLMOmniARSampling": "AR Sampling Params",
    "VLLMOmniDiffusionSampling": "Diffusion Sampling Params",
    "VLLMOmniSamplingParamsList": "Multi-Stage Sampling Params List",
}

WEB_DIRECTORY = "./web"
