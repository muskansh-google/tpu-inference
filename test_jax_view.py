import jax.numpy as jnp
import torch
import numpy as np
from torchax.interop import jax_view, torch_view

a = np.ones((2, 2))
b = torch_view(jnp.asarray(a))
print(f"torch_view type: {type(b)}")

c = jax_view(b)
print(f"jax_view(torch_view(x)) type: {type(c)}")

