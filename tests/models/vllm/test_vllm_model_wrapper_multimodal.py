
import unittest
from unittest.mock import MagicMock, patch
import torch
import jax
import jax.numpy as jnp
from typing import Any, Optional

# Mock vllm modules if not available for test running in specific envs
import sys
from types import SimpleNamespace

# Assume vllm is installed for running tests properly, but import defensively
try:
    from vllm.config import VllmConfig
    from tpu_inference.models.vllm.vllm_model_wrapper import VllmModelWrapper, _VllmRunner
except ImportError:
    # If imports fail (e.g. in CI without vLLM), skip or mock heavily
    # This is a placeholder for the user to run in a proper environment
    pass

class TestVllmModelWrapperMultimodal(unittest.TestCase):
    
    def setUp(self):
        # Determine if we can run
        try:
            import vllm
        except ImportError:
            self.skipTest("vllm not installed")

    @patch("tpu_inference.models.vllm.vllm_model_wrapper.get_tpu_quantization_config")
    @patch("tpu_inference.models.vllm.vllm_model_wrapper.VllmModelWrapper._apply_pp_patch")
    @patch("tpu_inference.models.vllm.vllm_model_wrapper.MultiHeadLatentAttentionWrapper")
    @patch("tpu_inference.models.vllm.vllm_model_wrapper.shard_model_to_tpu")
    def test_jit_embed_multimodal_func_call_structure(self, mock_shard, mock_mla, mock_patch, mock_quant):
        # Mock dependencies
        vllm_config = MagicMock()
        vllm_config.model_config = MagicMock()
        vllm_config.quant_config = MagicMock()
        vllm_config.compilation_config = MagicMock()
        vllm_config.load_config = MagicMock()
        vllm_config.load_config.load_format = "dummy"
        vllm_config.lora_config = None
        
        rng = jax.random.PRNGKey(0)
        mesh = MagicMock()
        
        # Initialize wrapper
        # We need to mock VllmModelWrapper.load_weights logic or bypass init issues
        # But we can just instantiate it if we mocked enough
        wrapper = VllmModelWrapper(vllm_config, rng, mesh)
        
        # Manually set .model to a mocked _VllmRunner
        mock_runner = MagicMock()
        wrapper.model = mock_runner
        
        # Get the JIT function
        embed_fn = wrapper.jit_embed_multimodal_fn()
        
        # Mock logic inside: torch.func.functional_call
        # We need to run the JIT function. 
        # Since we can't easily run JAX JIT with mocks in a non-TPU environment without some setup,
        # we might rely on checking the source or run it in CPU mode.
        try:
            jax.config.update("jax_platform_name", "cpu")
        except:
            pass
        
        # Params
        params = {"p1": jnp.array([1.0])}
        
        # Inputs
        image_grid = ((1, 2, 3),)
        video_grid = ((4, 5, 6),)
        grid = ((7, 8, 9),)
        
        # Kwargs (JAX arrays)
        pixel_values = jnp.array([0.1, 0.2])
        kwarg_keys = ("pixel_values",)
        
        # We need to patch torch.func.functional_call inside the module where VllmModelWrapper is defined
        with patch("torch.func.functional_call") as mock_func_call:
            mock_func_call.return_value = torch.tensor([1.0]) # Mock output

            # Call the JIT function
            # Note: We need to pass params as a Pytree.
            # And inputs.
            # JAX will trace this.
            
            # If we run this, JAX will trace. 
            # Inside the JIT, it calls functional_call.
            # We verify that functional_call was called with correct KWARGS.
            
            # Executing with split kwargs
            output = embed_fn(params, kwarg_keys, (pixel_values,), image_grid_thw=image_grid, video_grid_thw=video_grid, grid_thw=grid)
            
            # Verification
            # functional_call(module, params, args, kwargs)
            self.assertTrue(mock_func_call.called)
            
            call_args = mock_func_call.call_args
            # call_args[0] are positional: (module, params, args) or (module, params, kwargs=..., args=...) depending on signature
            # torch.func.functional_call(module, params, args=(), kwargs=None, ...)
            
            # Check module
            self.assertEqual(call_args[0][0], mock_runner)
            
            # Check kwargs passed to functional_call
            # call_args.kwargs might contain 'kwargs' key
            if 'kwargs' in call_args[1]:
                func_kwargs = call_args[1]['kwargs']
            else:
                # it might be passed as positional argument index 3?
                # functional_call(module, params_dict, args, kwargs)
                # module=0, params=1, args=2, kwargs=3
                if len(call_args[0]) > 3:
                     func_kwargs = call_args[0][3]
                else:
                     func_kwargs = {} # Should fail if empty

            # Check grid args presence in kwargs
            self.assertIn("image_grid_thw", func_kwargs)
            self.assertIn("video_grid_thw", func_kwargs)
            self.assertIn("grid_thw", func_kwargs)
            
            # Check values (should be tensors)
            self.assertTrue(torch.is_tensor(func_kwargs["image_grid_thw"]))
            # verify content of tensor matches input tuple
            # tensor from tuple ((1,2,3),) -> tensor([[1,2,3]])
            # torch.equal(..., torch.tensor(image_grid))
            # Note: mocking wrapper might complicate this check if we don't have functional_call return mocked correctly
            # But the inputs to functional_call should be tensors converted from inputs.
            
            self.assertIn("pixel_values", func_kwargs)
            # pixel_values should be a tensor (converted from JAX array)
            # Check that runner_method is passed
            self.assertIn("runner_method", func_kwargs)
            self.assertEqual(func_kwargs["runner_method"], "embed_multimodal")

            self.assertIn("pixel_values", func_kwargs)
            # pixel_values should be a tensor (converted from JAX array)
            self.assertTrue(torch.is_tensor(func_kwargs["pixel_values"]))

    @patch("tpu_inference.models.vllm.vllm_model_wrapper.get_tpu_quantization_config")
    @patch("tpu_inference.models.vllm.vllm_model_wrapper.VllmModelWrapper._apply_pp_patch")
    @patch("tpu_inference.models.vllm.vllm_model_wrapper.MultiHeadLatentAttentionWrapper")
    @patch("tpu_inference.models.vllm.vllm_model_wrapper.shard_model_to_tpu")
    def test_jit_embed_input_ids_fn(self, mock_shard, mock_mla, mock_patch, mock_quant):
        # Setup similar to above
        vllm_config = MagicMock()
        vllm_config.load_config.load_format = "dummy"
        wrapper = VllmModelWrapper(vllm_config, jax.random.PRNGKey(0), MagicMock())
        mock_runner = MagicMock()
        wrapper.model = mock_runner
        
        embed_fn = wrapper.jit_embed_input_ids_fn()
        
        params = {"p": jnp.array([1.0])}
        input_ids = jnp.array([1, 2, 3])
        
        with patch("torch.func.functional_call") as mock_func_call:
            mock_func_call.return_value = torch.tensor([1.0])
            
            output = embed_fn(params, input_ids)
            
            self.assertTrue(mock_func_call.called)
            # Args: (module, params, (input_ids_tensor,), {})
            call_args = mock_func_call.call_args
            self.assertEqual(call_args[0][0], mock_runner)
            
            # Check args
            # args_passed should occupy index 2 in call_args[0] if present as tuple, but here we use kwargs
            # functional_call(module, params, args=(), kwargs={...})
            if 'kwargs' in call_args[1]:
                func_kwargs = call_args[1]['kwargs']
            elif len(call_args[0]) > 3:
                func_kwargs = call_args[0][3]
            else:
                # args passed as positional 3rd argument? NO, we changed to kwargs only in wrapper
                func_kwargs = {}

            self.assertIn("input_ids", func_kwargs)
            self.assertTrue(torch.is_tensor(func_kwargs["input_ids"]))
            self.assertIn("runner_method", func_kwargs)
            self.assertEqual(func_kwargs["runner_method"], "embed_input_ids")



    def test_precompile_vision_encoder(self):
        vllm_config = MagicMock()
        vllm_config.load_config.load_format = "dummy"
        wrapper = VllmModelWrapper(vllm_config, None, MagicMock())
        
        # Should run without error
        wrapper.precompile_vision_encoder(MagicMock())

    def test_get_mrope_input_positions(self):
        vllm_config = MagicMock()
        vllm_config.load_config.load_format = "dummy"
        wrapper = VllmModelWrapper(vllm_config, None, MagicMock())
        mock_vllm_model = MagicMock()
        wrapper.vllm_model = mock_vllm_model
        
        # Test delegation
        wrapper.get_mrope_input_positions(input_tokens=[1], mm_features=[])
        mock_vllm_model.get_mrope_input_positions.assert_called_once()
        
        # Test fallback
        wrapper.vllm_model = MagicMock(spec=[]) # No get_mrope_input_positions, spec=[] prevents auto-creation of attrs
        with self.assertRaises(NotImplementedError):
            wrapper.get_mrope_input_positions(input_tokens=[1], mm_features=[])


if __name__ == "__main__":
    unittest.main()
