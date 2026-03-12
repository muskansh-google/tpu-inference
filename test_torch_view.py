import jax.numpy as jnp
import numpy as np
import torch
import torchax
from torchax.interop import torch_view

arr = np.array([1, 2, 3])
jarr = jnp.asarray(arr)
tval = torch_view(jarr)
print("Type of tval:", type(tval))
print("Isinstance of torch.Tensor?", isinstance(tval, torch.Tensor))

with torchax.default_env():
    tval2 = torch_view(jarr)
    print("Type of tval2 inside env:", type(tval2))
    print("Isinstance of torch.Tensor inside env?", isinstance(tval2, torch.Tensor))
