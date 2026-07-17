# vLLM Toolbox Integration
#
# This directory contains companion services and integration glue
# that connects the 1bit-systems NPU/GPU kernels with the 
# kyuz0/amd-strix-halo-vllm-toolboxes container.
#
# Files:
#   npu-companion.py       — NPU draft model server for speculative decoding
#   npu-companion.service  — systemd unit (auto-start inside container)
#   npu_accel.py           — vLLM launcher integration (--npu-draft flag)
#   Dockerfile.npu         — Multi-stage container build for NPU companion
#
# Usage (inside the vLLM toolbox container):
#   python3 npu-companion.py --model qwen3:0.6b
#
# Then launch vLLM with speculative decoding:
#   start-vllm --npu-draft
