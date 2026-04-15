# Qwen3-Coder-30B-A3B-Instruct (bf16) on 8 GPUs (FP8 has quant alignment issues at any TP)
PROFILE_NAME="qwen3-coder-30b-tp8"
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
MODEL_SHORT="qwen3-coder-30b"
TP_SIZE=8
MAX_MODEL_LEN=40960
OPENCODE_CONTEXT=32768
MAX_TOKENS=8192
GPU_MEMORY_UTIL=0.90
TOOL_CALL_PARSER=qwen3_xml
VLLM_EXTRA_ARGS="--max-num-seqs 6"
