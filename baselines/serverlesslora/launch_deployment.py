#!/usr/bin/env python3
"""Cluster launcher for ServerlessLoRA.

Orchestrates startup/shutdown of all components in dependency order:
  1. base_model_server.py  (one per GPU)  — TCP socket check, 600s timeout
  2. preload_scheduler.py  (global)       — HTTP /health check
  3. preload_agent.py      (one per GPU)  — HTTP /health check
  4. controller.py         (global)       — HTTP /health, then register functions+nodes

Usage:
    python launch_deployment.py --config deployment_config.yaml          # start + block
    python launch_deployment.py --config deployment_config.yaml --no-wait # start + exit
    python launch_deployment.py --config deployment_config.yaml --status  # check health
    python launch_deployment.py --config deployment_config.yaml --shutdown # kill all
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


class ClusterLauncher:
    """Manages lifecycle of all ServerlessLoRA components."""

    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.log_dir = os.path.join(PROJECT_ROOT, self.cfg.get("log_dir", "logs"))
        os.makedirs(self.log_dir, exist_ok=True)

        # Track all launched processes: list of (name, Popen)
        self._procs: list = []
        self._log_handles: list = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _open_log(self, name: str):
        path = os.path.join(self.log_dir, f"{name}.log")
        fh = open(path, "w")
        self._log_handles.append(fh)
        return fh

    @staticmethod
    def _gpu_env(device_id: int) -> dict:
        """Build per-GPU MPS environment variables.

        Each GPU has its own MPS daemon at /tmp/mps_{device_id}.
        The client always sees CUDA_VISIBLE_DEVICES=0 because the MPS
        server already remapped the physical GPU.
        """
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"
        env["CUDA_MPS_PIPE_DIRECTORY"] = f"/tmp/mps_{device_id}"
        return env

    def _launch_local(self, name: str, cmd: list, env: dict = None):
        """Launch a local subprocess, logging stdout+stderr."""
        log_fh = self._open_log(name)
        print(f"  Starting {name}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            preexec_fn=os.setsid,
            env=env,
        )
        self._procs.append((name, proc))
        return proc

    def _wait_tcp(self, host: str, port: int, timeout: float, label: str):
        """Wait until a TCP port is accepting connections."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with socket.create_connection((host, port), timeout=2):
                    print(f"  [OK] {label} ready on {host}:{port}")
                    return True
            except (ConnectionRefusedError, OSError, socket.timeout):
                time.sleep(2)
        print(f"  [FAIL] {label} not ready after {timeout}s")
        return False

    def _wait_http(self, url: str, timeout: float, label: str):
        """Wait until an HTTP endpoint responds 200."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(url, timeout=3)
                if r.status_code == 200:
                    print(f"  [OK] {label} healthy at {url}")
                    return True
            except requests.ConnectionError:
                pass
            time.sleep(2)
        print(f"  [FAIL] {label} not healthy after {timeout}s")
        return False

    def _post_json(self, url: str, data: dict, label: str, timeout: float = 60):
        """POST JSON and report result."""
        try:
            r = requests.post(url, json=data, timeout=timeout)
            resp = r.json()
            ok = resp.get("success", False)
            print(f"  [{'OK' if ok else 'FAIL'}] {label}: {resp}")
            return ok
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
            return False

    # ------------------------------------------------------------------
    # Launch stages
    # ------------------------------------------------------------------

    def _launch_base_model_servers(self):
        """Stage 1: Launch base_model_server.py for each GPU."""
        print("\n=== Stage 1: Base Model Servers ===")
        model_key = self.cfg.get("base_model", "llama-3.1-8b")
        # Resolve HF model ID from config.py's SUPPORTED_MODELS
        from config import SUPPORTED_MODELS
        model_id = SUPPORTED_MODELS.get(model_key, model_key)

        servers = []
        hostname = self.cfg["hostname"]
        for gpu in self.cfg["gpus"]:
            dev = gpu["device_id"]
            port = gpu["base_model_server_port"]
            name = f"bms_gpu{dev}"
            cmd = [
                sys.executable, "base_model_server.py",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--model", model_id,
                "--device", "cuda:0",
            ]
            self._launch_local(name, cmd, env=self._gpu_env(dev))
            servers.append((hostname, port, name))

        # Health check: TCP connect with 600s timeout (model loading is slow)
        print("\n  Waiting for base model servers (up to 600s per server)...")
        for host, port, name in servers:
            if not self._wait_tcp(host, port, 600, name):
                print(f"  WARNING: {name} failed to start, continuing anyway")

    def _launch_preload_scheduler(self):
        """Stage 2: Launch preload_scheduler.py (global, single instance)."""
        print("\n=== Stage 2: Preload Scheduler ===")
        sched_cfg = self.cfg.get("preload_scheduler", {})
        host = sched_cfg.get("host", "localhost")
        port = sched_cfg.get("port", 7100)
        interval = sched_cfg.get("interval_ms", 1000)

        cmd = [
            sys.executable, "preload_scheduler.py",
            "--port", str(port),
            "--interval-ms", str(interval),
        ]
        self._launch_local("preload_scheduler", cmd)
        self._wait_http(f"http://{host}:{port}/health", 60, "preload_scheduler")

    def _launch_preload_agents(self):
        """Stage 3: Launch preload_agent.py (one per GPU)."""
        print("\n=== Stage 3: Preload Agents ===")
        # Point agents to the controller (which hosts the scheduler with
        # the populated registry) instead of the standalone scheduler
        ctrl_cfg = self.cfg.get("controller", {})
        scheduler_url = f"http://{ctrl_cfg.get('host', 'localhost')}:{ctrl_cfg.get('port', 8000)}"
        defaults = self.cfg.get("defaults", {})
        keep_alive = defaults.get("keep_alive_ms", 60000)

        # Batch scheduling params from config
        batch_slo_ms = defaults.get("slo_ms", 2000.0)
        batch_max_batch_size = defaults.get("max_batch_size", 8)
        batch_base_ttft_ms = defaults.get("base_ttft_ms", 400.0)
        batch_marginal_cost_ms = defaults.get("marginal_cost_ms", 50.0)

        agents = []
        hostname = self.cfg["hostname"]
        for gpu in self.cfg["gpus"]:
            dev = gpu["device_id"]
            agent_port = gpu["agent_port"]
            bms_port = gpu["base_model_server_port"]
            worker_base = gpu["worker_base_port"]
            name = f"agent_gpu{dev}"
            cmd = [
                sys.executable, "preload_agent.py",
                "--node-id", f"gpu{dev}",
                "--hostname", hostname,
                "--port", str(agent_port),
                "--scheduler-url", scheduler_url,
                "--server-port", str(bms_port),
                "--worker-base-port", str(worker_base),
                "--device", "cuda:0",
                "--keep-alive-ms", str(keep_alive),
                "--batch-slo-ms", str(batch_slo_ms),
                "--batch-max-batch-size", str(batch_max_batch_size),
                "--batch-base-ttft-ms", str(batch_base_ttft_ms),
                "--batch-marginal-cost-ms", str(batch_marginal_cost_ms),
            ]
            self._launch_local(name, cmd, env=self._gpu_env(dev))
            agents.append((hostname, agent_port, name))

        print("\n  Waiting for preload agents (up to 60s each)...")
        for host, port, name in agents:
            self._wait_http(f"http://{host}:{port}/health", 60, name)

    def _launch_controller(self):
        """Stage 4: Launch controller.py and register functions + nodes."""
        print("\n=== Stage 4: Controller ===")
        ctrl_cfg = self.cfg.get("controller", {})
        host = ctrl_cfg.get("host", "localhost")
        port = ctrl_cfg.get("port", 8000)
        enable_preload = ctrl_cfg.get("enable_preload", True)
        enable_offload = ctrl_cfg.get("enable_offload", False)
        max_workers_per_gpu = self.cfg.get("defaults", {}).get("max_workers_per_gpu", 20)
        cmd = [
            sys.executable, "controller.py",
            "--port", str(port),
            "--max-workers-per-gpu", str(max_workers_per_gpu),
        ]
        if enable_preload:
            cmd.append("--enable-preload")
        if enable_offload:
            cmd.append("--enable-offload")
        if ctrl_cfg.get("disable_spawn_on_demand", False):
            cmd.append("--disable-spawn-on-demand")
        cmd.extend(["--config", os.path.abspath(self.config_path)])

        self._launch_local("controller", cmd)
        if not self._wait_http(f"http://{host}:{port}/health", 60, "controller"):
            print("  ERROR: Controller failed to start")
            return

        base_url = f"http://{host}:{port}"

        # Use short model name for backbone_id (must match artifact_id "backbone_{name}")
        model_key = self.cfg.get("base_model", "llama-3.1-8b")

        # Register functions
        print("\n  Registering functions...")
        for fn in self.cfg.get("functions", []):
            self._post_json(f"{base_url}/register_function", {
                "function_id": fn["function_id"],
                "backbone_id": model_key,
                "adapter_id": fn["adapter"],
                "slo_ms": fn.get("slo_ms", 2000.0),
            }, f"register function {fn['function_id']}")

        # Register nodes (one per GPU agent)
        print("\n  Registering nodes...")
        hostname = self.cfg["hostname"]
        for gpu in self.cfg["gpus"]:
            node_label = f"gpu{gpu['device_id']}"
            self._post_json(f"{base_url}/register_node", {
                "node_id": node_label,
                "hostname": hostname,
                "agent_port": gpu["agent_port"],
                "gpu_memory_mb": self.cfg.get("gpu_memory_mb", 44400.0),
                "container_memory_mb": self.cfg.get("container_memory_mb", 32768.0),
                "gpu_device": f"cuda:{gpu['device_id']}",
            }, f"register node {node_label}")

        # NOTE: Container scaling moved to _scale_containers(), called after
        # agents are launched (agents must be up to handle /spawn requests).

    def _scale_containers(self):
        """Scale up containers for each function. Must run AFTER agents are launched."""
        print("\n=== Stage 4b: Scale Containers ===")
        ctrl_cfg = self.cfg.get("controller", {})
        host = ctrl_cfg.get("host", "localhost")
        port = ctrl_cfg.get("port", 8000)
        base_url = f"http://{host}:{port}"

        num_functions = len(self.cfg.get("functions", []))
        default_cpf = self.cfg.get("defaults", {}).get("containers_per_function", 1)
        total_gpus = len(self.cfg["gpus"])
        print(f"  Scaling up containers "
              f"({num_functions} functions, {total_gpus} GPUs, "
              f"default {default_cpf} container(s) per function)...")

        # Refresh node heartbeats once before scaling
        hostname = self.cfg["hostname"]
        for gpu in self.cfg["gpus"]:
            node_label = f"gpu{gpu['device_id']}"
            requests.post(f"{base_url}/register_node", json={
                "node_id": node_label,
                "hostname": hostname,
                "agent_port": gpu["agent_port"],
                "gpu_memory_mb": self.cfg.get("gpu_memory_mb", 44400.0),
                "container_memory_mb": self.cfg.get("container_memory_mb", 32768.0),
                "gpu_device": f"cuda:{gpu['device_id']}",
            }, timeout=5)

        # Scale all functions (per-function override via "containers" key)
        # All functions scale in parallel so controller handles total_containers
        # spawns concurrently — timeout must account for total cluster load.
        total_containers = sum(fn.get("containers", default_cpf)
                               for fn in self.cfg.get("functions", []))
        scale_timeout = max(120, total_containers * 3)

        def _scale_one(fn):
            count = fn.get("containers", default_cpf)
            self._post_json(f"{base_url}/scale", {
                "function_id": fn["function_id"],
                "target_count": count,
            }, f"scale {fn['function_id']} -> {count}", timeout=scale_timeout)

        # Scale all functions in parallel — agent-side semaphore (4 concurrent
        # per GPU) prevents thundering herd while maximizing throughput
        with ThreadPoolExecutor(max_workers=num_functions) as executor:
            futures = [executor.submit(_scale_one, fn)
                       for fn in self.cfg.get("functions", [])]
            for f in as_completed(futures):
                try:
                    f.result()
                except Exception as e:
                    print(f"  Scale error: {e}")

        # Wait for ALL containers to become ready (not just 5s sleep)
        expected = sum(fn.get("containers", default_cpf)
                       for fn in self.cfg.get("functions", []))
        max_wait = 300  # 5 minutes
        poll_interval = 5
        elapsed = 0
        ready = 0
        print(f"\n  Waiting for {expected} containers to be ready...")

        while elapsed < max_wait:
            try:
                r = requests.get(f"{base_url}/containers", timeout=5)
                containers = r.json().get("containers", [])
                ready = sum(1 for c in containers if c.get("status") == "ready")
                if ready >= expected:
                    print(f"  All {ready} containers ready ({elapsed}s).")
                    for c in containers:
                        print(f"    {c.get('container_id', 'unknown')}: "
                              f"{c.get('function_id', '?')} "
                              f"status={c.get('status', '?')}")
                    return
                print(f"  Containers ready: {ready}/{expected} ({elapsed}s)")
            except Exception:
                pass
            time.sleep(poll_interval)
            elapsed += poll_interval

        # Timed out — print what we have
        print(f"  WARNING: Only {ready}/{expected} containers ready "
              f"after {max_wait}s, proceeding anyway.")
        try:
            r = requests.get(f"{base_url}/containers", timeout=5)
            for c in r.json().get("containers", []):
                print(f"    {c.get('container_id', 'unknown')}: "
                      f"{c.get('function_id', '?')} "
                      f"status={c.get('status', '?')}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Profiling (Section 4.2)
    # ------------------------------------------------------------------

    def _run_profiler(self):
        """Stage 5: Profile workers to calibrate T₀, α, and SLO per function.

        Paper Section 4.2: "The Offline Profiler runs during system deployment
        to measure T_0 and α for each registered function."

        Paper Section 6.1: "We set the TTFT SLO as 5× the first warm-start
        invocation's TTFT."
        """
        print("\n=== Stage 5: Offline Profiling ===")

        ctrl_cfg = self.cfg.get("controller", {})
        ctrl_host = ctrl_cfg.get("host", "localhost")
        ctrl_port = ctrl_cfg.get("port", 8000)
        ctrl_url = f"http://{ctrl_host}:{ctrl_port}"

        defaults = self.cfg.get("defaults", {})
        profile_batch_sizes = defaults.get("profile_batch_sizes", [1, 2, 4, 8])
        profile_warmup = defaults.get("profile_warmup_runs", 2)
        profile_samples = defaults.get("profile_sample_runs", 3)
        profile_max_tokens = defaults.get("profile_max_tokens", 32)
        profile_prompt_tokens = defaults.get("profile_prompt_tokens", 128)

        # Discover containers per function from controller
        try:
            r = requests.get(f"{ctrl_url}/containers", timeout=10)
            containers = r.json().get("containers", [])
        except Exception as e:
            print(f"  WARNING: Cannot fetch containers: {e}")
            print("  Skipping profiling, using default parameters.")
            return

        if not containers:
            print("  No containers available, skipping profiling.")
            return

        # Build function -> first reachable container URL
        func_workers = {}
        for c in containers:
            fid = c.get("function_id", "")
            if fid and fid not in func_workers and c.get("status") == "ready":
                port = c.get("http_port")
                if port:
                    url = f"http://localhost:{port}"
                    try:
                        requests.get(f"{url}/health", timeout=3)
                        func_workers[fid] = url
                    except Exception:
                        pass  # skip unreachable containers

        if not func_workers:
            print("  No ready containers found, skipping profiling.")
            return

        # Build a profiling prompt of the target input length.
        # "word " is 1-2 tokens for most tokenizers; we over-generate
        # slightly and let the tokenizer truncate.
        profile_prompt = "word " * profile_prompt_tokens
        print(f"  Profiling {len(func_workers)} function(s) "
              f"with batch sizes {profile_batch_sizes}, "
              f"~{profile_prompt_tokens} input tokens")

        # Enable profiling mode on ALL ready containers so the batch
        # scheduler dispatches immediately (no fill-or-expire delay).
        # This prevents queue wait from polluting TTFT measurements.
        print("  Enabling profiling mode on all containers...")
        for c in containers:
            port = c.get("http_port")
            if port and c.get("status") == "ready":
                try:
                    requests.post(
                        f"http://localhost:{port}/set_profiling_mode",
                        json={"enabled": True}, timeout=5,
                    )
                except Exception:
                    pass

        # Import profiler
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
        from profiler import OfflineProfiler, save_profiles

        profiler = OfflineProfiler(
            batch_sizes=profile_batch_sizes,
            warmup_runs=profile_warmup,
            sample_runs=profile_samples,
            max_tokens=profile_max_tokens,
        )

        profiles = []
        try:
            for fid, worker_url in func_workers.items():
                print(f"\n  Profiling {fid} at {worker_url}...")
                try:
                    result = profiler.profile_function(fid, worker_url, prompt=profile_prompt)
                    profiles.append(result)
                    print(f"    T_0 = {result.base_ttft_ms:.1f}ms, "
                          f"α = {result.marginal_cost_ms:.1f}ms, "
                          f"R² = {result.r_squared:.4f}")
                except Exception as e:
                    print(f"    FAILED: {e}")
        finally:
            # Always disable profiling mode when done, even on error
            print("  Disabling profiling mode on all containers...")
            for c in containers:
                port = c.get("http_port")
                if port:
                    try:
                        requests.post(
                            f"http://localhost:{port}/set_profiling_mode",
                            json={"enabled": False}, timeout=5,
                        )
                    except Exception:
                        pass

        if not profiles:
            print("  No profiles generated, using defaults.")
            return

        # Save profiles to disk
        profile_path = os.path.join(self.log_dir, "profiles.json")
        save_profiles(profiles, profile_path)
        print(f"\n  Saved {len(profiles)} profiles to {profile_path}")

        # Push calibrated parameters to controller
        # Paper Section 6.1: SLO = 5 × warm-start TTFT (= T₀)
        # But never go below the config-level SLO floor (e.g. 2000ms).
        config_slo = defaults.get("slo_ms", 2000.0)
        print("  Pushing calibrated parameters to controller...")
        for p in profiles:
            calibrated_slo = max(config_slo, 5.0 * p.base_ttft_ms)

            try:
                r = requests.post(f"{ctrl_url}/update_profile", json={
                    "function_id": p.function_id,
                    "base_ttft_ms": p.base_ttft_ms,
                    "marginal_cost_ms": p.marginal_cost_ms,
                    "slo_ms": calibrated_slo,
                }, timeout=10)
                resp = r.json()
                ok = resp.get("success", False)
                print(f"    {p.function_id}: T_0={p.base_ttft_ms:.1f}ms, "
                      f"α={p.marginal_cost_ms:.1f}ms, "
                      f"SLO={calibrated_slo:.0f}ms "
                      f"[{'OK' if ok else 'FAIL'}]")
            except Exception as e:
                print(f"    {p.function_id}: FAILED to push profile: {e}")

        # Also push to each worker's batch scheduler via its /update_batch_config endpoint
        print("  Pushing batch config to workers...")
        for c in containers:
            fid = c.get("function_id", "")
            port = c.get("http_port")
            if not fid or not port:
                continue
            # Find matching profile
            matching = [p for p in profiles if p.function_id == fid]
            if not matching:
                continue
            p = matching[0]
            calibrated_slo = max(config_slo, 5.0 * p.base_ttft_ms)
            try:
                requests.post(f"http://localhost:{port}/update_batch_config", json={
                    "base_ttft_ms": p.base_ttft_ms,
                    "marginal_cost_ms": p.marginal_cost_ms,
                    "slo_ms": calibrated_slo,
                }, timeout=5)
            except Exception:
                pass  # Worker may not support this endpoint yet

        print("\n  Profiling complete.")

    def _load_cached_profiles(self):
        """Load previously saved profiles and push to controller/workers."""
        profile_path = os.path.join(self.log_dir, "profiles.json")
        if not os.path.exists(profile_path):
            print(f"\n=== Skipping profiling (no cached profiles at {profile_path}) ===")
            return

        print(f"\n=== Loading cached profiles from {profile_path} ===")
        import json
        with open(profile_path) as f:
            raw_data = json.load(f)
        raw_profiles = raw_data.get("profiles", raw_data) if isinstance(raw_data, dict) else raw_data

        ctrl_cfg = self.cfg.get("controller", {})
        ctrl_url = f"http://{ctrl_cfg.get('host', 'localhost')}:{ctrl_cfg.get('port', 8000)}"
        config_slo = self.cfg.get("defaults", {}).get("slo_ms", 2000.0)

        for p in raw_profiles:
            fid = p["function_id"]
            base_ttft = p["base_ttft_ms"]
            marginal_cost = p["marginal_cost_ms"]
            calibrated_slo = max(config_slo, 5.0 * base_ttft)
            try:
                r = requests.post(f"{ctrl_url}/update_profile", json={
                    "function_id": fid,
                    "base_ttft_ms": base_ttft,
                    "marginal_cost_ms": marginal_cost,
                    "slo_ms": calibrated_slo,
                }, timeout=10)
                ok = r.json().get("success", False)
                print(f"  {fid}: T_0={base_ttft:.1f}ms, α={marginal_cost:.1f}ms, "
                      f"SLO={calibrated_slo:.0f}ms [{'OK' if ok else 'FAIL'}]")
            except Exception as e:
                print(f"  {fid}: FAILED to push profile: {e}")

        # Push to workers too
        try:
            r = requests.get(f"{ctrl_url}/containers", timeout=10)
            containers = r.json().get("containers", [])
        except Exception:
            containers = []

        for c in containers:
            fid = c.get("function_id", "")
            port = c.get("http_port")
            if not fid or not port:
                continue
            matching = [p for p in raw_profiles if p["function_id"] == fid]
            if not matching:
                continue
            p = matching[0]
            calibrated_slo = max(config_slo, 5.0 * p["base_ttft_ms"])
            try:
                requests.post(f"http://localhost:{port}/update_batch_config", json={
                    "base_ttft_ms": p["base_ttft_ms"],
                    "marginal_cost_ms": p["marginal_cost_ms"],
                    "slo_ms": calibrated_slo,
                }, timeout=5)
            except Exception:
                pass

        print(f"  Loaded {len(raw_profiles)} cached profiles.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def launch(self, skip_profiling: bool = False):
        """Full startup sequence in dependency order."""
        print(f"Launching cluster: {self.cfg.get('cluster_name', 'unnamed')}")
        self._launch_base_model_servers()
        # Controller must start before agents, since agents poll the
        # controller's /decisions/<node_id> endpoint.
        self._launch_controller()
        self._launch_preload_agents()
        self._scale_containers()

        if not skip_profiling:
            self._run_profiler()
        else:
            self._load_cached_profiles()

        print("\n=== Cluster launch complete ===")
        print(f"  Components running: {len(self._procs)}")
        print(f"  Logs directory: {self.log_dir}")

    def shutdown(self):
        """Reverse-order SIGTERM, then SIGKILL after 5s grace."""
        # Callers (run_throughput.sh / run_trace_driven.sh) already announce
        # this step with a timestamp; avoid a redundant untimestamped repeat.

        # Also try to hit controller /shutdown endpoint
        ctrl_cfg = self.cfg.get("controller", {})
        ctrl_url = f"http://{ctrl_cfg.get('host', 'localhost')}:{ctrl_cfg.get('port', 8000)}"
        try:
            requests.post(f"{ctrl_url}/shutdown", timeout=3)
        except Exception:
            pass

        # Reverse order: controller, agents, scheduler, base model servers
        for name, proc in reversed(self._procs):
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                print(f"  SIGTERM -> {name} (pid {proc.pid})")
            except (ProcessLookupError, OSError):
                pass

        # Grace period
        time.sleep(5)

        # SIGKILL remaining
        for name, proc in reversed(self._procs):
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    print(f"  SIGKILL -> {name} (pid {proc.pid})")
                except (ProcessLookupError, OSError):
                    pass

        # Close log handles
        for fh in self._log_handles:
            try:
                fh.close()
            except Exception:
                pass

        self._procs.clear()
        self._log_handles.clear()

        # Clean up GPU contention shared memory segments
        try:
            from utils.gpu_contention import GPUContentionTracker
            for gpu in self.cfg.get("gpus", []):
                dev = gpu["device_id"]
                GPUContentionTracker.cleanup(dev)
                print(f"  Cleaned up GPU contention tracker for gpu:{dev}")
        except Exception as e:
            print(f"  GPU contention cleanup: {e}")

        print("Shutdown complete.")

    def status(self):
        """Query cluster health by checking all endpoints."""
        print("=== Cluster Status ===\n")

        # Controller
        ctrl_cfg = self.cfg.get("controller", {})
        ctrl_host = ctrl_cfg.get("host", "localhost")
        ctrl_port = ctrl_cfg.get("port", 8000)
        try:
            r = requests.get(
                f"http://{ctrl_host}:{ctrl_port}/status", timeout=5
            )
            data = r.json()
            print(f"Controller ({ctrl_host}:{ctrl_port}): UP")
            print(f"  Total requests: {data.get('total_requests', 0)}")
            print(f"  Success rate: {data.get('success_rate', 0):.1%}")
            print(f"  Avg E2E: {data.get('avg_e2e_ms', 0):.1f}ms")
            print(f"  Avg TTFT: {data.get('avg_ttft_ms', 0):.1f}ms")
        except Exception as e:
            print(f"Controller ({ctrl_host}:{ctrl_port}): DOWN ({e})")

        # Scheduler is hosted inside the controller; no standalone check needed.
        print(f"\nScheduler: hosted inside controller (no standalone process)")

        # Agents and BMS per GPU
        hostname = self.cfg["hostname"]
        print(f"\nHost: {hostname}")
        for gpu in self.cfg["gpus"]:
            dev = gpu["device_id"]
            agent_port = gpu["agent_port"]
            try:
                r = requests.get(
                    f"http://{hostname}:{agent_port}/health", timeout=3
                )
                data = r.json()
                print(f"  GPU {dev} Agent (:{agent_port}): {data.get('status', 'unknown')}")
            except Exception:
                print(f"  GPU {dev} Agent (:{agent_port}): DOWN")

            # Base model server (TCP)
            bms_port = gpu["base_model_server_port"]
            try:
                with socket.create_connection((hostname, bms_port), timeout=2):
                    print(f"  GPU {dev} BMS   (:{bms_port}): UP")
            except Exception:
                print(f"  GPU {dev} BMS   (:{bms_port}): DOWN")


def main():
    parser = argparse.ArgumentParser(
        description="ServerlessLoRA Cluster Launcher"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to deployment_config.yaml",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Start cluster and exit (don't block)",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Check health of all components",
    )
    parser.add_argument(
        "--shutdown", action="store_true",
        help="Shutdown all components",
    )
    parser.add_argument(
        "--skip-profiling", action="store_true",
        help="Skip offline profiling stage (use default T_0, alpha, SLO)",
    )
    args = parser.parse_args()

    launcher = ClusterLauncher(args.config)

    if args.status:
        launcher.status()
        return

    if args.shutdown:
        # For shutdown without prior launch, we just hit endpoints and
        # try to kill any matching processes
        launcher.shutdown()
        return

    # Normal launch
    launcher.launch(skip_profiling=args.skip_profiling)

    if args.no_wait:
        # Print PIDs so user can manage them
        print("\nRunning processes:")
        for name, proc in launcher._procs:
            print(f"  {name}: pid {proc.pid}")
        print("\nUse --shutdown to stop all components.")
        return

    # Block until Ctrl-C
    print("\nCluster running. Press Ctrl-C to shutdown.")
    try:
        while True:
            # Check if any process has died
            for name, proc in launcher._procs:
                if proc.poll() is not None:
                    print(f"\n  WARNING: {name} exited with code {proc.returncode}")
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n\nReceived Ctrl-C, shutting down...")
        launcher.shutdown()


if __name__ == "__main__":
    main()
