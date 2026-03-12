import jax.numpy as jnp
import torch
from torchax import torch_view
import numpy as np

a = np.ones((2, 2))
b = torch_view(jnp.asarray(a))
print(f"torch_view type: {type(b)}")

try:
    c = jnp.asarray(b)
    print(f"jnp.asarray(torch_view) type: {type(c)}")
except Exception as e:
    print(f"Error: {e}")

