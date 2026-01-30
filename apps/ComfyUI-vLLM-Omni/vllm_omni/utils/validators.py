from .models import MODEL_PIPELINE_SPECS


def validate_sampling_params_types(model_name, sampling_param_list: dict | list[dict] | None = None):
    pipeline_spec = MODEL_PIPELINE_SPECS.get(model_name)
    if pipeline_spec is None:
        return

    stages = pipeline_spec["stages"]
    if isinstance(stages, str):
        stages = [stages]
    if isinstance(sampling_param_list, list):
        if len(stages) != len(sampling_param_list):
            raise ValueError(
                f"Sampling parameter list length {len(sampling_param_list)} does not match number of stages {len(stages)} for model {model_name}."
            )
        for i, sp in enumerate(sampling_param_list):
            if sp["type"] != stages[i]:
                raise ValueError(
                    f"Sampling parameter type ({sp['type']}) does not match stage type ({stages[i]}) at index {i} for model {model_name}."
                )
    elif isinstance(sampling_param_list, dict):
        if any(stage != sampling_param_list["type"] for stage in stages):
            raise ValueError(
                f"When passing a single sampling parameter node, all stages of the model must match this sampling parameter type ({sampling_param_list['type']})."
                f"However, the stages of model {model_name} are: {stages}"
            )


def validate_model_config(model_name, image=None, audio=None, video=None, sampling_param_list: dict | list[dict] | None = None):
    """Validate inputs and raise errors if wrong. Don't return anything if the input passes. Don't check if the model is not in the specs (to allow custom models)"""
    pipeline_spec = MODEL_PIPELINE_SPECS.get(model_name)
    if pipeline_spec is None:
        return

    required_inputs = pipeline_spec["required_inputs"]
    provided_inputs = []
    if image is not None:
        provided_inputs.append("image")
    if audio is not None:
        provided_inputs.append("audio")
    if video is not None:
        provided_inputs.append("video")

    for required_input in required_inputs:
        if required_input not in provided_inputs:
            raise ValueError(f"Model {model_name} requires input: {required_input}")


def add_sampling_parameters_to_stage(
    model_name: str, sampling_param_list: dict | list[dict] | None, stage_type: str, /, **params_to_add
) -> dict | list[dict]:
    """Given a model's name and the sampling parameter list to query this model, add arbitrary additional parameters to the sampling parameters of all stages of the given type."""
    pipeline_spec = MODEL_PIPELINE_SPECS.get(model_name)
    if not pipeline_spec:
        print(
            f"Since the model {model_name} is not in our list, we cannot ensure if the fields ({tuple(params_to_add.keys())}) are added to the correct stage's sampling params. We will do it heuristiclly."
        )
        pipeline_spec = {"stages": ["diffusion"]}

    stages = pipeline_spec["stages"]
    if isinstance(sampling_param_list, dict):
        sampling_param_list.update(params_to_add)
    elif sampling_param_list is None:
        sampling_param_list = params_to_add.copy()
    else:
        for i, stage in enumerate(stages):
            if stage == stage_type:
                sampling_param_list[i].update(params_to_add)

    return sampling_param_list
