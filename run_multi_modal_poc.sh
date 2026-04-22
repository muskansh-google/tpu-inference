#!/bin/bash
set -e

# Configuration
VM_NAME="muskansh-v6e-1"
ZONE="us-central2-b"
PROJECT="tpu-prod-env-one-vm"
REMOTE_DIR="~/tpu-inference_poc"
LOCAL_DIR="/usr/local/google/home/muskansh/tpu-inference"

echo "=== Syncing workspace to $VM_NAME ==="
# Create remote dir if not exists (should already exist from previous step, but safe)
gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "mkdir -p $REMOTE_DIR"

# Sync tpu_inference and examples directories
gcloud compute tpus tpu-vm scp --recurse "$LOCAL_DIR/tpu_inference" "$VM_NAME:$REMOTE_DIR/" --zone "$ZONE" --project "$PROJECT"
gcloud compute tpus tpu-vm scp --recurse "$LOCAL_DIR/examples" "$VM_NAME:$REMOTE_DIR/" --zone "$ZONE" --project "$PROJECT"

echo "=== Running Multi Modal Inference on $VM_NAME ==="
CMD="cd $REMOTE_DIR && MODEL_IMPL_TYPE=vllm PYTHONPATH=$REMOTE_DIR ~/vllm_env/bin/python examples/multi_modal_inference.py \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --test-multi-image \
  --max-model-len 8192 --gpu-memory-utilization 0.85 2>&1 | tee output.log"
gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "$CMD"
