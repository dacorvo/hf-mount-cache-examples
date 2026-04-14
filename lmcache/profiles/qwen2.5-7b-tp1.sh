# Qwen2.5-Coder-7B-Instruct on 1 GPU (custom tool parser for function calling)
PROFILE_NAME="qwen2.5-7b-tp1"
MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_SHORT="qwen2.5-coder-7b"
TP_SIZE=1
MAX_MODEL_LEN=32768
MAX_TOKENS=4096
GPU_MEMORY_UTIL=0.90
TOOL_CALL_PARSER=qwen2_5_coder
TOOL_PARSER_PLUGIN="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/lib/qwen2_5_coder_tool_parser.py"
CHAT_TEMPLATE="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}/lib/tool_chat_template_qwen2_5_coder.jinja"
