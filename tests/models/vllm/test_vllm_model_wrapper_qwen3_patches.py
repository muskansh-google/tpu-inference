# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest
import torch

from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper


def test_apply_patches_criteria_not_qwen():
    fake_self = MagicMock()
    fake_self.vllm_config.model_config.hf_config.model_type = "llama"
    fake_self.vllm_config.load_config.model_loader_extra_config = {
        "use_deepstack": True
    }

    vllm_model = MagicMock()
    fake_self.model.vllm_model = vllm_model

    orig_embed = MagicMock()
    vllm_model.embed_input_ids = orig_embed

    VllmModelWrapper._apply_qwen3_vl_patches(fake_self)

    # Verify NOT applied
    assert vllm_model.embed_input_ids == orig_embed


def test_apply_patches_criteria_deepstack_false():
    fake_self = MagicMock()
    fake_self.vllm_config.model_config.hf_config.model_type = "qwen"
    fake_self.vllm_config.load_config.model_loader_extra_config = {
        "use_deepstack": False
    }

    vllm_model = MagicMock()
    fake_self.model.vllm_model = vllm_model

    orig_embed = MagicMock()
    vllm_model.embed_input_ids = orig_embed

    VllmModelWrapper._apply_qwen3_vl_patches(fake_self)

    # Verify NOT applied
    assert vllm_model.embed_input_ids == orig_embed


def test_apply_patches_success():
    fake_self = MagicMock()
    fake_self.vllm_config.model_config.hf_config.model_type = "qwen"
    fake_self.vllm_config.load_config.model_loader_extra_config = {
        "use_deepstack": True
    }

    vllm_model = MagicMock()
    vllm_model.use_deepstack = True
    vllm_model.visual_dim = 16

    # original methods
    orig_embed = MagicMock()
    orig_forward = MagicMock()
    vllm_model.embed_input_ids = orig_embed
    vllm_model.forward = orig_forward

    fake_self.model.vllm_model = vllm_model

    VllmModelWrapper._apply_qwen3_vl_patches(fake_self)

    # Verify patches applied (they should be new functions)
    assert vllm_model.embed_input_ids != orig_embed
    assert vllm_model.forward != orig_forward


def test_packing_embeds():
    fake_self = MagicMock()
    fake_self.vllm_config.model_config.hf_config.model_type = "qwen"
    fake_self.vllm_config.load_config.model_loader_extra_config = {
        "use_deepstack": True
    }

    vllm_model = MagicMock()
    vllm_model.use_deepstack = True
    vllm_model.visual_dim = 4
    vllm_model.embed_input_ids = MagicMock(return_value=torch.ones(2, 4))

    deepstack_embeds = torch.ones(2, 2, 4) * 2  # (seq=2, num_layers=2, dim=4)
    vllm_model.deepstack_input_embeds = deepstack_embeds

    fake_self.model.vllm_model = vllm_model

    VllmModelWrapper._apply_qwen3_vl_patches(fake_self)

    patched_embed = vllm_model.embed_input_ids

    result = patched_embed()

    # Original is (2, 4)
    # Deepstack packed is (2, 2*4) = (2, 8)
    # Cat dim=-1 results in (2, 4 + 8) = (2, 12)
    assert result.shape == (2, 12)
    # First 4 are ones
    assert torch.allclose(result[:, :4], torch.ones(2, 4))
    # Next 8 are twos
    assert torch.allclose(result[:, 4:], torch.ones(2, 8) * 2)


def test_unpacking_embeds():
    fake_self = MagicMock()
    fake_self.vllm_config.model_config.hf_config.model_type = "qwen"
    fake_self.vllm_config.load_config.model_loader_extra_config = {
        "use_deepstack": True
    }

    vllm_model = MagicMock()
    vllm_model.use_deepstack = True
    vllm_model.visual_dim = 4
    vllm_model.deepstack_num_level = 2
    vllm_model.config.vision_config.deepstack_visual_indexes = [1, 2]

    orig_forward = MagicMock()
    vllm_model.forward = orig_forward

    mock_set_deepstack = MagicMock()
    vllm_model._set_deepstack_input_embeds = mock_set_deepstack

    fake_self.model.vllm_model = vllm_model

    with patch(
        "tpu_inference.models.vllm.vllm_model_wrapper.get_pp_group"
    ) as mock_pp:
        mock_pp.return_value.is_first_rank = True

        VllmModelWrapper._apply_qwen3_vl_patches(fake_self)

        patched_forward = vllm_model.forward

        # Packed input (seq=2, packed_dim=12)
        packed_input = torch.ones(2, 12)
        # First 4 are text embeds, next 8 are deepstack
        packed_input[:, 4:] = 2

        patched_forward(
            input_ids="dummy",
            positions="dummy",
            intermediate_tensors="dummy",
            inputs_embeds=packed_input,
        )

        # Verify orig_forward called with unpacked embeds
        orig_forward.assert_called_once()
        called_kwargs = orig_forward.call_args[1]
        assert called_kwargs["inputs_embeds"].shape == (2, 4)
        assert torch.allclose(called_kwargs["inputs_embeds"], torch.ones(2, 4))

        # Verify _set_deepstack_input_embeds called
        mock_set_deepstack.assert_called_once()
        called_dict = mock_set_deepstack.call_args[0][0]
        assert "deepstack_input_embeds_1" in called_dict
        assert "deepstack_input_embeds_2" in called_dict
        assert called_dict["deepstack_input_embeds_1"].shape == (2, 4)
        assert torch.allclose(
            called_dict["deepstack_input_embeds_1"], torch.ones(2, 4) * 2
        )
