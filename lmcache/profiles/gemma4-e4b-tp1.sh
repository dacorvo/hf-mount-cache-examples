# Gemma-4-E4B-it on 1 GPU (standard attention, LMCache compatible)
PROFILE_NAME="gemma4-e4b-tp1"
MODEL="google/gemma-4-E4B-it"
MODEL_SHORT="gemma4-e4b"
TP_SIZE=1
MAX_MODEL_LEN=32768
MAX_TOKENS=4096
GPU_MEMORY_UTIL=0.90
TOOL_CALL_PARSER=gemma4
REASONING_PARSER=gemma4
CHAT_TEMPLATE="$SCRIPT_DIR/lib/tool_chat_template_gemma4.jinja"
LIMIT_MM_PER_PROMPT='{"image":0,"audio":0}'
