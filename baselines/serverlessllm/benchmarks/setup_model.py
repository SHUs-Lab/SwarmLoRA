#!/usr/bin/env python3
"""
Setup script for registering a base model with LoRA adapters on ServerlessLLM.

Usage:
    python benchmarks/setup_model.py \
        --server-url http://127.0.0.1:8343 \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --num-adapters 10 \
        --adapter-path-template "/models/adapters/adapter_{i}" \
        --num-gpus 1 --max-instances 10 --wait-ready
"""

import argparse
import json
import sys
import time

import requests


def parse_args():
    parser = argparse.ArgumentParser(
        description="Register a base model with LoRA adapters on ServerlessLLM"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8343",
        help="ServerlessLLM head node URL (default: http://127.0.0.1:8343)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model name to register",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers", "dummy"],
        help="Backend to use (default: transformers)",
    )
    parser.add_argument(
        "--num-adapters",
        type=int,
        default=10,
        help="Number of LoRA adapters to register (0 through N-1)",
    )
    parser.add_argument(
        "--adapter-path-template",
        type=str,
        default="./models/raw_adapters/adapter_{i}",
        help="Path template for adapter directories. Use {i} as placeholder for adapter index.",
    )
    parser.add_argument(
        "--adapter-name-template",
        type=str,
        default="adapter_{i}",
        help="Name template for adapters. Use {i} as placeholder for index.",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=None,
        help="Number of GPUs per model instance (supports fractional, e.g., 0.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Internal batch size for TransformersBackend (1=disabled)",
    )
    parser.add_argument(
        "--batch-wait-ms",
        type=int,
        default=150,
        help="Max ms to wait for more requests before processing batch",
    )
    parser.add_argument(
        "--mode",
        choices=["original", "batched", "optimized"],
        default="original",
        help="Deployment mode: original (no batching, 1 GPU/instance), "
             "batched (batching only, 1 GPU/instance), "
             "optimized (batching + 0.5 GPU/instance)",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=10,
        help="Maximum auto-scaling instances",
    )
    parser.add_argument(
        "--min-instances",
        type=int,
        default=None,
        help="Minimum (pre-warmed) instances. Default: 0 for original, max_instances for batched/optimized",
    )
    parser.add_argument(
        "--target-concurrency",
        type=int,
        default=1,
        help="Auto-scaling concurrency target per instance",
    )
    parser.add_argument(
        "--keep-alive",
        type=int,
        default=0,
        help="Keep-alive seconds for idle instances (0 = scale to zero)",
    )
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        default=True,
        help="Poll /v1/models until model appears (default: True)",
    )
    parser.add_argument(
        "--no-wait-ready",
        action="store_false",
        dest="wait_ready",
        help="Don't wait for model to be ready after registration",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Max seconds to wait for model readiness",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print registration payload without sending",
    )
    return parser.parse_args()


def build_registration_payload(args):
    """Build the JSON payload for POST /register."""
    lora_adapters = {}
    for i in range(args.num_adapters):
        adapter_name = args.adapter_name_template.replace("{i}", str(i))
        adapter_path = args.adapter_path_template.replace("{i}", str(i))
        lora_adapters[adapter_name] = adapter_path

    if args.backend == "dummy":
        backend_config = {}
    else:
        backend_config = {
            "pretrained_model_name_or_path": args.model_name,
            "torch_dtype": "float16",
            "hf_model_class": "AutoModelForCausalLM",
            "device_map": "auto",
            "enable_lora": True,
            "lora_adapters": lora_adapters,
        }
        # Batching config
        if args.batch_size > 1:
            backend_config["batch_size"] = args.batch_size
            backend_config["batch_wait_ms"] = args.batch_wait_ms
            backend_config["max_concurrency"] = 100

    payload = {
        "model": args.model_name,
        "backend": args.backend,
        "num_gpus": args.num_gpus,
        "backend_config": backend_config,
        "auto_scaling_config": {
            "metric": "concurrency",
            "target": args.target_concurrency,
            "min_instances": args.min_instances,
            "max_instances": args.max_instances,
            "keep_alive": args.keep_alive,
        },
    }
    return payload


def register_model(server_url, payload):
    """Register the model via POST /register.

    /register blocks server-side until the model is converted to sllm_store's
    format, which is a one-time ~35min cost on first registration (fast
    reload thereafter). The request timeout here must outlast that or a slow
    first run gets reported as a registration failure -- even though the
    server is still healthy and the conversion completes in the background.
    """
    url = f"{server_url.rstrip('/')}/register"
    try:
        response = requests.post(
            url, json=payload, headers={"Content-Type": "application/json"}, timeout=3600
        )
        if response.status_code == 200:
            print(f"Model '{payload['model']}' registered successfully.")
            return True
        else:
            print(
                f"Registration failed: {response.status_code} {response.text}",
                file=sys.stderr,
            )
            return False
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}", file=sys.stderr)
        return False


def wait_for_model(server_url, model_name, timeout):
    """Poll GET /v1/models until the model appears."""
    url = f"{server_url.rstrip('/')}/v1/models"
    start = time.time()
    print(f"Waiting for model '{model_name}' to be ready (timeout: {timeout}s)...")

    while time.time() - start < timeout:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data if isinstance(data, list) else data.get("models", [])
                model_ids = []
                for m in models:
                    if isinstance(m, dict):
                        model_ids.append(m.get("id", m.get("model", "")))
                    elif isinstance(m, str):
                        model_ids.append(m)
                if model_name in model_ids:
                    elapsed = time.time() - start
                    print(f"Model '{model_name}' is ready ({elapsed:.1f}s).")
                    return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)

    print(f"Timeout ({timeout}s) waiting for model '{model_name}'.", file=sys.stderr)
    return False


def apply_mode_defaults(args):
    """Apply mode presets, allowing explicit CLI flags to override."""
    if args.mode == "original":
        if args.num_gpus is None:
            args.num_gpus = 1.0
        if args.batch_size is None:
            args.batch_size = 1
        if args.target_concurrency == 1:
            pass  # keep default
        if args.min_instances is None:
            args.min_instances = 0
    elif args.mode == "batched":
        if args.num_gpus is None:
            args.num_gpus = 1.0
        if args.batch_size is None:
            args.batch_size = 16
        if args.target_concurrency == 1:
            args.target_concurrency = 4
        if args.min_instances is None:
            args.min_instances = args.max_instances
        if args.keep_alive == 0:
            args.keep_alive = 300
    elif args.mode == "optimized":
        if args.num_gpus is None:
            args.num_gpus = 0.5
        if args.batch_size is None:
            args.batch_size = 16
        if args.target_concurrency == 1:
            args.target_concurrency = 4
        if args.min_instances is None:
            args.min_instances = args.max_instances
        if args.keep_alive == 0:
            args.keep_alive = 300

    # Fallback defaults
    if args.num_gpus is None:
        args.num_gpus = 1.0
    if args.batch_size is None:
        args.batch_size = 1
    if args.min_instances is None:
        args.min_instances = 0


def main():
    args = parse_args()
    apply_mode_defaults(args)
    payload = build_registration_payload(args)

    print("Registration payload:")
    print(json.dumps(payload, indent=2))
    print()

    if args.dry_run:
        print("[Dry run] No request sent.")
        return

    success = register_model(args.server_url, payload)
    if not success:
        sys.exit(1)

    if args.wait_ready:
        ready = wait_for_model(args.server_url, args.model_name, args.timeout)
        if not ready:
            sys.exit(1)

    print("Setup complete.")


if __name__ == "__main__":
    main()
