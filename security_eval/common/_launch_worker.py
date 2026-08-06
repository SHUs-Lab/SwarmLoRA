"""Launch one SwarmLoRA worker process (its own OS process, one LoRA adapter + KV cache)."""

import argparse
import os
import sys

# project root on path so the in-tree .so extensions resolve (worker_sync adds cwd)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, "src"))
os.chdir(_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--http-port", type=int, required=True)
    ap.add_argument("--agg-host", default="localhost")
    ap.add_argument("--agg-port", type=int, default=50056)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--lora", required=True)
    args = ap.parse_args()

    from worker.worker_sync import FaaSWorker
    w = FaaSWorker(host=args.agg_host, port=args.agg_port,
                   device=args.device, lora_id=args.lora)
    w.initialize()
    print(f"[worker {args.http_port}] initialized lora={args.lora} "
          f"worker_id={w.worker_id}; serving", flush=True)
    w.start_http_server(args.http_port)


if __name__ == "__main__":
    main()
