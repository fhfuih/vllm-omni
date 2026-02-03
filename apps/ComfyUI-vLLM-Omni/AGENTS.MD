# 🧠 ComfyUI Plugin for vLLM-Omni: Multimodal Generation Nodes

This project implements a ComfyUI plugin that provides custom nodes for interacting with **vLLM-Omni**, an all-in-one multimodal generative AI hosting service. It supports both diffusion and autoregression models via an OpenAI-compatible API.

---

## 🚀 Overview

The plugin enables users to generate images, videos, and audio using hosted pipelines from vLLM-Omni. These pipelines may involve multiple stages of either diffusion or autoregression models, each requiring specific sampling parameters. The plugin abstracts this complexity into modular, reusable ComfyUI nodes.

### 🔗 Communication
All communication is done through the **OpenAI-compatible API** exposed by vLLM-Omni. No local model loading or inference is performed.

### 🧩 Modalities Supported
- **Text-to-Image**
- **Text-to-Video**
- **Image-to-Image**
- **Video-to-Video**
- (More combinations supported depending on pipeline)

> ⚠️ Note: Models do not output text-only responses. Every pipeline must produce at least one non-text modality (image/audio/video).

---

## 🛠️ Node Architecture

### 1. **Main Generation Nodes**

Each main node corresponds to a modality and handles generation requests.

#### Available Nodes:
| Node Name           | Output Modality | Optional Inputs              |
|---------------------|------------------|------------------------------|
| `Generate Image`    | Image            | Image, Audio, Video, Sampling Params |
| `Generate Video`    | Video            | Audio, Image, Sampling Params |
| `Generate Audio`    | Audio            | Image, Audio, Video, Sampling Params |

#### Common Inputs (Widgets):
- `URL` *(string)* – Endpoint of the vLLM-Omni service.
- `Model` *(string)* – Name of the pipeline to use.
- `Prompt` *(multiline text)* – Input prompt for generation.
- `Negative Prompt` *(multiline text)* – Optional negative guidance.

#### Modality-Specific Inputs:
- **Image & Video Nodes**:
  - `Width` *(int)*
  - `Height` *(int)*
- **Video Node Only**:
  - `Num Frames` *(int)*
  - `FPS` *(int)*

#### Outputs:
Each node has two output connections:
1. **Generated content** (Image / Video / Audio)
2. **Text response** (if any metadata or debug info is returned)

---

### 2. **Sampling Parameter Nodes**

Since pipelines can be multi-stage and require different parameter sets per stage, we define modular sampling parameter nodes.

#### Available Nodes:
| Node Type                     | For Which Model Stage? | Fields                                                                 |
|------------------------------|------------------------|------------------------------------------------------------------------|
| `AR Sampling`    | Autoregression stage   | `max_tokens`, `temperature`, `top_p`, `repetition_penalty`            |
| `Diffusion Sampling`         | Diffusion stage        | `num_inference_steps`, `guidance_scale`                               |
| `Sampling Params List` | Aggregate 1–3 stages    | Accepts up to three sampling param nodes; outputs aggregated config   |

##### Aggregation Behavior:
- Takes **1 required** and **2 more optional** sampling param nodes as inputs.
- Produces a single aggregated configuration for the main generation node.

> 💡 Tip: A main node should accept either an individual sampling parameter node or an aggregator node.

---

## 🧪 Input Validation System

Every model in vLLM-Omni maps to a predefined pipeline with known modality requirements and expected number/type of sampling parameter sets.

### Validation Routines (Architectural Support)

Each model name is mapped to a validation routine that checks:

1. ✅ Whether the provided input modalities are compatible with the model.
2. ✅ Whether the correct number of sampling parameter sets are provided (if the model input is a sampling parameter aggregator node).
3. ✅ Whether each sampling parameter set matches the type expected by the corresponding stage.

> 📌 AI only need to generate placeholder routines in the codebase.

---

## 🧱 Tech Stack

- **Framework**: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) – Custom Node API
- **API Client**: `openai` Python package (OpenAI-compatible endpoint)
- **Validation Engine**: Internal mapping + pluggable validator functions

> 📘 Refer to MCP named `comfyui-doc` for detailed ComfyUI node development documentation.

---

## 📁 Project Structure (Suggested)

```
project root
├── __init__.py
├── vllm_omni
│   ├── __init__.py
│   ├── nodes.py               # Generate nodes
│   └── utils
│       ├── api_client.py      # Wrapper around openai client
│       ├── format.py          # Utility function for media file format conversion
│       ├── models.py          # Models' pipeline specs and other metadata
│       └── validators.py      # Model-specific validation logic
└── web                        # JavaScript for optional ComfyUI frontend
```

---

## 🧩 Example Workflow (Conceptual)

1. User selects `Generate Image` node.
2. Connects:
   - One `Autoregression Sampling` node
   - One `Diffusion Sampling` node
   - Feeds both into a `Sampling Params Aggregator`
3. Connects aggregator to the main node.
4. Sets `URL`, `Model`, `Prompt`, and optional `Negative Prompt`.
5. Executes workflow → receives generated image + optional text response.

---

## 📌 Summary

This plugin bridges ComfyUI workflows with powerful multimodal AI pipelines hosted on vLLM-Omni. By encapsulating complex multi-stage generation logic into intuitive nodes, it empowers creators to experiment freely while maintaining performance and scalability.

---

Let the multimodal magic flow! 🌈✨
