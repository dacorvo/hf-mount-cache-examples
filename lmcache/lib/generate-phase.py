#!/usr/bin/env python3
"""Generate a process-compose YAML from the Jinja2 template.

Usage: generate-phase.py [--mount|--mount-overlay] <phase> <prompt-dir> <output>

Reads configuration from environment variables, discovers prompt files
from the given directory, and renders templates/phase.yaml.j2.
"""

import argparse
import glob
import os
import sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def build_vllm_command() -> str:
    """Build the vLLM serve command as a single string."""
    parts = [
        f"vllm serve {os.environ['MODEL']}",
        f"--port {os.environ['VLLM_PORT']}",
        f"--max-model-len {os.environ['MAX_MODEL_LEN']}",
        f"--gpu-memory-utilization {os.environ['GPU_MEMORY_UTIL']}",
        """--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'""",
    ]
    extra = os.environ.get("VLLM_EXTRA_ARGS", "").strip()
    if extra:
        parts.append(extra)
    parts.extend([
        "--enable-auto-tool-choice",
        f"--tool-call-parser {os.environ['TOOL_CALL_PARSER']}",
    ])
    return " ".join(parts)


def discover_conversations(prompt_dir: str) -> list[dict]:
    """Find all .txt prompt files and return as a list of {name, file}."""
    files = sorted(glob.glob(os.path.join(prompt_dir, "*.txt")))
    if not files:
        print(f"ERROR: no prompt files found in {prompt_dir}", file=sys.stderr)
        sys.exit(1)
    return [
        {"name": Path(f).stem, "file": f}
        for f in files
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount", action="store_const", const="rw", dest="mount_mode")
    parser.add_argument("--mount-overlay", action="store_const", const="overlay", dest="mount_mode")
    parser.add_argument("phase")
    parser.add_argument("prompt_dir")
    parser.add_argument("output")
    args = parser.parse_args()

    script_dir = os.environ["SCRIPT_DIR"]
    template_dir = os.path.join(script_dir, "templates")

    conversations = discover_conversations(args.prompt_dir)

    env = Environment(
        loader=FileSystemLoader(template_dir),
        keep_trailing_newline=True,
    )
    template = env.get_template("phase.yaml.j2")

    ctx = {
        "phase": args.phase,
        "mount_mode": args.mount_mode,
        "conversations": conversations,
        "vllm_command": build_vllm_command(),
        "vllm_port": os.environ["VLLM_PORT"],
        "max_model_len": os.environ["MAX_MODEL_LEN"],
        "log_dir": os.environ["LOG_DIR"],
        "script_dir": script_dir,
        "vllm_url": os.environ["VLLM_URL"],
        "virtual_env": os.environ["VIRTUAL_ENV"],
        "hf_token": os.environ.get("HF_TOKEN", ""),
        "mount_point": os.environ["MOUNT_POINT"],
        "profile_name": os.environ["PROFILE_NAME"],
        "cache_dir": os.environ["CACHE_DIR"],
        "lmcache_config_file": os.environ["LMCACHE_CONFIG_FILE"],
        "hf_mount_bin": os.environ.get("HF_MOUNT_BIN", ""),
        "hf_mount_cache_dir": os.environ.get("HF_MOUNT_CACHE_DIR", ""),
        "bucket": os.environ.get("BUCKET", ""),
    }

    rendered = template.render(**ctx)
    Path(args.output).write_text(rendered)

    print(f"Generated: {args.output} ({len(conversations)} conversations, mount={args.mount_mode or 'none'})")


if __name__ == "__main__":
    main()
