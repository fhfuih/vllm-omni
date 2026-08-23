# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def test_diffusion_kv_metadata_uses_native_cache_group_block_ids() -> None:
    sequence = DiffusionKVSequenceMetadata(
        sequence_id=1,
        prefix_len=4,
        target_len=2,
        seq_len=8,
        block_ids=([1, 2], [5, 6]),
    )
    metadata = DiffusionKVMetadata(
        request_id="req-0",
        allocation_generation=3,
        sequences=(sequence,),
    )

    assert metadata.sequences[0].block_ids == ((1, 2), (5, 6))
