import unittest
from unittest.mock import MagicMock, patch
import jax.numpy as jnp
from tpu_inference.runner.multimodal_manager import MultiModalManager, MultiModalConfig

class TestQwen3VLSupport(unittest.TestCase):
    def setUp(self):
        self.mock_runner = MagicMock()
        self.mock_runner.model_config.multimodal_config = MultiModalConfig(limit_per_prompt={"image": 1, "video": 1})
        self.manager = MultiModalManager(self.mock_runner)

    def test_execute_mm_encoder_converts_video_grid_thw(self):
        # Setup inputs
        # Simulate video input requiring grid_thw processing
        batched_mm_inputs = {
            "pixel_values_videos": jnp.zeros((1, 3, 16, 224, 224)),
            "video_grid_thw": jnp.array([[1, 14, 14]]) # Shape (1, 3)
        }
        image_grid_thw = () # No images

        # Mock embed_multimodal_fn
        self.mock_runner.embed_multimodal_fn = MagicMock(return_value=jnp.zeros((1, 1024)))

        # Run execute_mm_encoder
        self.manager.execute_mm_encoder(
            image_grid_thw=image_grid_thw,
            video_grid_thw=None, # Passed inside batched_mm_inputs usually or separate depending on call site, 
                                 # but execute_mm_encoder signature has video_grid_thw kwarg?
                                 # Checking signature: def execute_mm_encoder(self, image_grid_thw, ..., **batched_mm_inputs)
                                 # Actually `video_grid_thw` will be in `batched_mm_inputs` if passed from `forward`.
                                 # But `MultiModalManager.execute_mm_encoder` signature might need verification.
            **batched_mm_inputs
        )
        
        # Verify call args
        # The key verification is that video_grid_thw was converted to tuple of tuples
        call_args = self.mock_runner.embed_multimodal_fn.call_args
        _, kwargs = call_args
        
        self.assertIn("video_grid_thw", kwargs if "video_grid_thw" in kwargs else call_args[0]) 
        # Actually based on my edit:
        # self.runner.embed_multimodal_fn(self.runner.state, image_grid_thw, video_grid_thw, **batched_mm_inputs)
        
        # Extract args
        args = call_args[0] # state, image_grid_thw, video_grid_thw
        video_grid_arg = args[2]
        
        self.assertIsInstance(video_grid_arg, tuple)
        self.assertEqual(video_grid_arg, ((1, 14, 14),))

if __name__ == '__main__':
    unittest.main()
