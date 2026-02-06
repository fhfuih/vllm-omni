#!/usr/bin/env python

"""Tests for `scaffold_plugin` package."""

import pytest
from comfyui_vllm_omni.nodes import VLLMOmniSamplingParamsList


@pytest.fixture
def example_sp_list_node():
    """Fixture to create an Example node instance."""
    return VLLMOmniSamplingParamsList()


def test_example_node_initialization(example_sp_list_node):
    """Test that the node can be instantiated."""
    assert isinstance(example_sp_list_node, VLLMOmniSamplingParamsList)


def test_return_types():
    """Test the node's metadata."""
    assert VLLMOmniSamplingParamsList.RETURN_TYPES == ("SAMPLING_PARAMS",)
    assert VLLMOmniSamplingParamsList.FUNCTION == "aggregate"
    assert VLLMOmniSamplingParamsList.CATEGORY == "vLLM-Omni/Sampling Params"
