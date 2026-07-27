# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    ModulateIndexPrepare,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_modulate_index_is_cached_by_shape_and_device() -> None:
    prepare = ModulateIndexPrepare(zero_cond_t=True)
    timestep = torch.tensor([1.0])
    img_shapes = [[(1, 2, 3), (1, 1, 2)]]

    doubled_timestep, first = prepare(timestep, img_shapes)
    _, second = prepare(timestep, img_shapes)

    torch.testing.assert_close(doubled_timestep, torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(first, torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1]], dtype=torch.int))
    assert second is first

    _, different_shape = prepare(timestep, [[(1, 2, 2), (1, 1, 2)]])
    assert different_shape is not first


def test_modulate_index_cache_is_bounded() -> None:
    prepare = ModulateIndexPrepare(zero_cond_t=True)
    timestep = torch.tensor([1.0])
    first_shapes = [[(1, 1, 1), (1, 1, 1)]]

    _, first = prepare(timestep, first_shapes)
    for width in range(2, 18):
        prepare(timestep, [[(1, 1, width), (1, 1, 1)]])

    assert len(prepare._modulate_index_cache) == 16

    _, recreated = prepare(timestep, first_shapes)
    assert recreated is not first
    assert len(prepare._modulate_index_cache) == 16


def test_modulate_index_is_not_created_without_zero_conditioning() -> None:
    prepare = ModulateIndexPrepare(zero_cond_t=False)
    timestep = torch.tensor([1.0])

    returned_timestep, modulate_index = prepare(timestep, [[(1, 1, 1)]])

    assert returned_timestep is timestep
    assert modulate_index is None
    assert not prepare._modulate_index_cache
