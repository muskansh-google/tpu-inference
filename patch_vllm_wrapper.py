import re

with open("tpu_inference/models/vllm/vllm_model_wrapper.py", "r") as f:
    text = f.read()

replacement = """
            for k, v in kwargs.items():
                torch_kwargs[k] = _convert_to_torchax(v)
                print(f"DEBUG_CONVERT: {k} before={type(v)} after={type(torch_kwargs[k])}")
"""
text = text.replace("            for k, v in kwargs.items():\n                torch_kwargs[k] = _convert_to_torchax(v)", replacement.strip("\n"))

with open("tpu_inference/models/vllm/vllm_model_wrapper.py", "w") as f:
    f.write(text)
