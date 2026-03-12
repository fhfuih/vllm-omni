"""
Offline inference tests: text-to-image.
See examples/offline_inference/text_to_image/README.md
"""

from pathlib import Path

import pytest

from tests.conftest import assert_image_valid
from tests.examples.conftest import EXAMPLES, ExampleRunner, ReadmeSnippet
from tests.utils import hardware_marks

pytestmark = [pytest.mark.advanced_model, pytest.mark.example, *hardware_marks(res={"cuda": "H100"})]

T2I_SCRIPT = EXAMPLES / "offline_inference" / "text_to_image" / "text_to_image.py"
README_PATH = T2I_SCRIPT.with_name("README.md")
EXAMPLE_OUTPUT_SUBFOLDER = "example_offline_t2i"
SKIPPED_H2_SECTIONS = {"LoRA", "Web UI Demo"}


README_SNIPPETS = ReadmeSnippet.extract_readme_snippets(README_PATH)


@pytest.mark.parametrize("snippet", README_SNIPPETS, ids=lambda snippet: snippet.test_id)
def test_text_to_image(snippet: ReadmeSnippet, example_runner: ExampleRunner):
    if snippet.h2_title in SKIPPED_H2_SECTIONS:
        pytest.skip(f"README section '{snippet.h2_title}' is intentionally excluded for examples tests")

    result = example_runner.run(snippet, output_subfolder=Path(EXAMPLE_OUTPUT_SUBFOLDER))
    for asset in result.assets:
        assert_image_valid(asset)
