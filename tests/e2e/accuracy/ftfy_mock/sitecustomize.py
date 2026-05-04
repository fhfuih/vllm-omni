"""
This sitecustomize.py is used to ensure the presence of an ftfy implementation.
ftfy is a text encoding sanitizer and it is implicitly required by diffusers' WanImageToVideoPipeline.
And this pipeline is used in several accuracy tests.

A sitecustomize.py is a Python mechanism to inject code into the interpreter at startup.
To use this sitecustomize.py, include its directory in the PYTHONPATH environment variable.

The sitecustomize approach must be used because diffusers' Wan pipelines may be imported in a subprocess.
For example, when launching with vLLM-Omni's diffusers backend in online serving mode.
`subprocess.Popen` (used by OmniServer test helper) does not inherit any mocks from the parent process.
This is the only way to inject mocks into the subprocess.

.. note::
   If installing the real ftfy library, the relevant tests may fail the similarity assertion.
   Because vLLM-Omni doesn't use ftfy to preprocess the text input.
   Hence, we must hack diffusers' Wan pipelines to not use ftfy either.
"""

import threading

print("[sitecustomize] started", flush=True)
threading.Timer(
    30.0,
    lambda: print("[sitecustomize] diffusers import still in progress after 30s", flush=True),
).start()
threading.Timer(
    120.0,
    lambda: print("[sitecustomize] diffusers import still in progress after 2 minutes", flush=True),
).start()


class _IdentityFtfy:
    @staticmethod
    def fix_text(text: str) -> str:
        return text


def _ensure_wan_ftfy_fallback() -> None:
    try:
        print("[sitecustomize] starting to pre-import diffusers.pipelines.wan.pipeline_wan_i2v", flush=True)
        from diffusers.pipelines.wan import pipeline_wan_i2v as wan_i2v_module

        print("[sitecustomize] finished importing diffusers.pipelines.wan.pipeline_wan_i2v", flush=True)
    except ImportError:
        print("[sitecustomize] diffusers import raised ImportError", flush=True)
        return

    if not hasattr(wan_i2v_module, "ftfy"):
        wan_i2v_module.ftfy = _IdentityFtfy()
        print("ftfy (text encoding sanitizer) is not installed. Using mock ftfy implementation (identity function)")
    else:
        print("ftfy (text encoding sanitizer) is installed. Using actual ftfy implementation.")


_ensure_wan_ftfy_fallback()
