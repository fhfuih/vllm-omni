# Structured data as interaction events during realtime video generation

Currently, realtime video generation interaction only supports prompt update. (API is at `vllm_omni/entrypoints/openai/serving_video_output_stream.py` and actual payload consumption is at `vllm_omni/diffusion/worker/diffusion_model_runner.py`.) This design doc adds structured data to the interaction payload.

## Input Protocol

### Submit One Interaction

The core request submits one interaction to take effect at the next eligible
model-generation boundary:

```json
{
  "type": "session.interaction",
  "interaction": {
    "interaction_id": "controller-update-42",
    "event": {
      "prompt": "A vehicle travels through a snowy mountain pass",
      "multi_modal_data": {
        "camera": {
          "schema": "camera.v1",
          "mode": "target",
          "data": {
            "... camera target fields ...": "..."
          }
        },
        "action": {
          "schema": "action.v1",
          "mode": "velocity",
          "data": {
            "... action velocity fields ...": "..."
          }
        }
      }
    },
    "transition_chunks": 2
  }
}
```

`event` is analogous to a subset of `OmniTextPrompt`:

- `prompt` is optional. When present, it is a target value.
- Also optional `negative_prompt` if the model supports updating it.
- `multi_modal_data` maps modality to structured payloads.
- A composite event may update several modalities atomically at the same
  effective boundary.

Each structured payload declares:

- `schema`: a versioned schema interpreted by a model adapter;
  - And one modality may have different representations. For example, camera transformation can be represented by a 4x4 matrix or two vectors of _translation_ and _rotation_.
- `mode`: `target` or `velocity` (explained below);
- `data`: schema-specific structural values.

### `target` Semantics

A `target` describes the state that a track should reach. It is the **default** mode

- The interaction may provide one **transition** control:
  - `transition_chunks`: model-generation chunks before the targets are reached.
  - `transition_seconds`: generated-media time before the targets are reached;
  - These fields are mutually exclusive.
  - If both are omitted, each target uses the model's default.
  - 0 means a hard change

- `transition_seconds` transition is rounded up to the nearest chunk boundary.
  - For example, provided `transition_seconds=2`, the model uses 33 frames per chunk, and the user initially set `fps=16`.
In this case, `transition_seconds` resolves to `2.0625`

- If a target replaces an active velocity on the same track, the velocity is
cancelled at the target's effective boundary. The target transition starts
from the state resolved at that boundary.

### Velocity Semantics

A velocity describes a transformation to be applied once every `1/fps` second.
It remains active until it expires or another input replaces the same track.

An update to another track does not cancel the velocity. A
replacement scheduled for the same track cancels it only when the replacement
becomes effective.

### Identity and Idempotency

`interaction_id` belongs to the individual interaction:

- the client may provide a session-unique string;
- otherwise, the server generates a time-ordered UUID and returns it;
  - UUID to prevent accidental collision with another client-set id

## Response Protocol

### Queue Acknowledgement and Errors

The core acknowledgement confirms queue admission, not model application:

```json
{
  "type": "session.interaction.queued",
  "request_id": "video_stream-...",
  "interaction_id": "controller-update-42"
}
```

Errors are correlated to the interaction when its identity is known:

```json
{
  "type": "error",
  "request_id": "video_stream-...",
  "interaction_id": "controller-update-42",
  "code": "unsupported_interaction_schema",
  "message": "The active model does not support camera.v1"
}
```

### Video Chunk Correlation

A metadata message **immediately precedes** each binary video payload:

```json
{
  "type": "video.chunk_metadata",
  "request_id": "video_stream-...",
  "kind": "media",
  "transport_chunk_index": 12,
  "generation_chunk_index": 10,
  "num_frames": 9,
  "byte_length": 184320,
  "started_interaction_ids": ["controller-update-42"],
  "active_interaction_ids": ["controller-update-42"],
  "completed_interaction_ids": []
}
```

- `transport_chunk_index` identifies the encoded payload, while `generation_chunk_index` identifies the model-generation unit.

- For example, MP4 Fragment (m4s) may contain trailer chunks, so these two indices may not always be the same.

- A trailer chunk may use `kind: "trailer"`, a null generation index, and zero frames.
