#!/usr/bin/env python3
"""
Start ServerlessLLM server for single-node benchmarking.

Requires: sllm/utils.py patched to not skip control_node in get_worker_nodes().
Start Ray with: ray start --head --num-gpus=2 --resources='{"control_node":1,"worker_id_0":1}'
"""

import ray
import uvicorn

from sllm.app_lib import create_app
from sllm.controller import SllmController


def main():
    host = "0.0.0.0"
    port = 8343

    if not ray.is_initialized():
        print("[*] Initializing Ray...")
        ray.init()
    else:
        print("[*] Ray already initialized")

    print("[*] Creating FastAPI application...")
    app = create_app()

    print("[*] Starting SLLM controller...")
    controller_cls = ray.remote(SllmController)
    controller = controller_cls.options(
        name="controller", num_cpus=1, resources={"control_node": 0.1}
    ).remote({
        "enable_storage_aware": False,
        "enable_migration": False,
    })

    ray.get(controller.start.remote())
    print("[OK] SLLM controller started successfully")

    print(f"[*] Starting server on {host}:{port}...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
