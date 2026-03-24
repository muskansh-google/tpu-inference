# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

import os
import sys

# Disable CUDA-specific shared experts stream for TPU
# This prevents errors when trying to create CUDA streams on TPU hardware
# The issue was introduced by vllm-project/vllm#26440
os.environ["VLLM_DISABLE_SHARED_EXPERTS_STREAM"] = "1"


def _alias_vllm_args():
    # Alias deprecated flags for backwards compatibility
    # --guided-decoding-backend -> --structured-outputs-config.backend
    # This must run before argparse.parse_args() in vllm.entrypoints.cli.main.
    # Since tpu_inference is imported when vllm.platforms.current_platform is
    # accessed (which happens during cli_env_setup() in main.py),
    # this script should run early enough.

    old_flag = "--guided-decoding-backend"
    new_flag = "--structured-outputs-config.backend"

    if not any(arg.startswith(old_flag) for arg in sys.argv):
        return

    new_argv = []
    for arg in sys.argv:
        if arg.startswith(old_flag + "="):
            new_argv.append(new_flag + "=" + arg[len(old_flag) + 1:])
        elif arg == old_flag:
            new_argv.append(new_flag)
        else:
            new_argv.append(arg)
    sys.argv[:] = new_argv


_alias_vllm_args()
