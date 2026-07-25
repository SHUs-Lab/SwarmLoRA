"""Live fault-isolation driver: BR / RT / RLR, system-agnostic.

  blast_radius      = co-tenants whose in-flight request failed / co-tenants
  request_loss_rate = failed in-flight requests / in-flight at fault
  recovery_time_ms  = ms from fault until a fresh request succeeds; -1 if never
"""

import argparse
import json
import os
import signal
import threading
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class FaultResult:
    system: str
    fault_type: str
    num_tenants: int
    blast_radius: float
    recovery_time_ms: float
    request_loss_rate: float
    inflight_at_fault: int = 0
    failed_requests: int = 0
    affected_cotenants: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Signal -> realistic worker-death cause it models:
#   SIGKILL  : Linux OOM-killer / supervisor `kill -9` (uncatchable abrupt death)
#   SIGSEGV  : segfault / memory corruption in a C/CUDA extension
#   SIGABRT  : CUDA-fatal error / C++ assertion failure (torch internal check)
_SIGNAL_MAP = {
    "kill":  signal.SIGKILL,
    "segv":  signal.SIGSEGV,
    "abort": signal.SIGABRT,
}


def _extract_tokens(payload):
    for k in ("tokens", "completion_tokens"):
        v = payload.get(k)
        if isinstance(v, int) and v > 0:
            return v
    # S-LoRA /generate returns text; approximate token count by whitespace split.
    txt = payload.get("generated_text") or payload.get("text")
    if isinstance(txt, list):
        txt = " ".join(str(x) for x in txt)
    if isinstance(txt, str) and txt.strip():
        return len(txt.split())
    return None


def _post(url, body, timeout):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
            ok = (r.status == 200) and bool(payload.get("success", True))
            return {"ok": ok, "status": r.status, "ms": (time.perf_counter() - t0) * 1000,
                    "tokens": _extract_tokens(payload)}
    except Exception as e:
        return {"ok": False, "status": None, "ms": (time.perf_counter() - t0) * 1000,
                "err": type(e).__name__, "tokens": None}


def run(tenants, victim_index, victim_pid, fault_type, system,
        fault_delay, req_timeout, recovery_timeout, recovery_url=None,
        fault_signal="kill"):
    """tenants: [{"id","url","payload"}]. Returns FaultResult."""
    n = len(tenants)
    results = [None] * n
    done = [False] * n

    def worker(i):
        results[i] = _post(tenants[i]["url"], tenants[i]["payload"], req_timeout)
        done[i] = True

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    t_launch = time.perf_counter()
    for th in threads:
        th.start()

    # Let everyone get mid-generation, then inject the fault.
    time.sleep(fault_delay)
    inflight_at_fault = sum(1 for d in done if not d)
    t_fault = time.perf_counter()
    sig = _SIGNAL_MAP.get(fault_signal, signal.SIGKILL)
    killed = False
    try:
        os.kill(victim_pid, sig)
        killed = True
    except ProcessLookupError:
        pass
    print(f"[fault] t={t_fault - t_launch:.2f}s {sig!r}({fault_signal}) pid={victim_pid} "
          f"(killed={killed}); in-flight at fault={inflight_at_fault}", flush=True)

    # Probe a surviving endpoint for recovery time.
    survivors = [t for j, t in enumerate(tenants) if j != victim_index]
    if recovery_url:
        probe_target = {"url": recovery_url, "payload": dict(tenants[victim_index]["payload"])}
    else:
        probe_target = survivors[0] if survivors else tenants[victim_index]
    probe_body = dict(probe_target["payload"])
    # shrink the probe so RT reflects availability, not generation length
    if "max_tokens" in probe_body:
        probe_body["max_tokens"] = 1
    if "parameters" in probe_body and isinstance(probe_body["parameters"], dict):
        probe_body["parameters"] = dict(probe_body["parameters"])
        probe_body["parameters"]["max_new_tokens"] = 1

    recovered = False
    recovery_time_ms = float("inf")
    t_rec_start = time.perf_counter()
    while time.perf_counter() - t_rec_start < recovery_timeout:
        r = _post(probe_target["url"], probe_body, min(req_timeout, 10))
        if r["ok"]:
            recovered = True
            recovery_time_ms = (time.perf_counter() - t_fault) * 1000
            break
        time.sleep(0.25)

    for th in threads:
        th.join(timeout=req_timeout + recovery_timeout + 5)

    # Exclude victim's own request from blast_radius (intended death, not collateral).
    victim_ok = bool(results[victim_index] and results[victim_index]["ok"])
    cotenant_results = [results[j] for j in range(n) if j != victim_index]
    affected = sum(1 for r in cotenant_results if not (r and r["ok"]))
    cotenants = max(1, n - 1)
    failed_total = sum(1 for r in results if not (r and r["ok"]))

    survivors_ok = [r for r in cotenant_results if r and r["ok"]]
    survivors_completed = sum(1 for r in survivors_ok if (r.get("tokens") or 0) > 0)
    survivor_tokens = [r.get("tokens") for r in survivors_ok if r.get("tokens")]
    min_survivor_tokens = min(survivor_tokens) if survivor_tokens else 0
    all_survivors_completed = (survivors_completed == len(cotenant_results))

    blast_radius = affected / cotenants
    request_loss_rate = (failed_total / inflight_at_fault) if inflight_at_fault else 0.0

    fr = FaultResult(
        system=system,
        fault_type=fault_type,
        num_tenants=n,
        blast_radius=round(blast_radius, 4),
        recovery_time_ms=(round(recovery_time_ms, 1) if recovered else -1.0),
        request_loss_rate=round(request_loss_rate, 4),
        inflight_at_fault=inflight_at_fault,
        failed_requests=failed_total,
        affected_cotenants=affected,
        params={"victim_index": victim_index, "victim_pid": victim_pid,
                "fault_signal": fault_signal,
                "victim_request_failed": (not victim_ok), "killed": killed,
                "recovered": recovered, "recovery_timeout_s": recovery_timeout,
                "cotenants_total": len(cotenant_results),
                "survivors_completed": survivors_completed,
                "all_survivors_completed": all_survivors_completed,
                "min_survivor_tokens": min_survivor_tokens,
                "per_tenant": results},
        note=("recovery_time_ms=-1 means the system did NOT serve a fresh request "
              "within recovery_timeout (no auto-recovery). survivors_completed counts "
              "co-tenants that returned 200 AND generated tokens."),
    )
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--fault-type", default="crash")
    ap.add_argument("--fault-signal", default="kill", choices=["kill", "segv", "abort"],
                    help="kill=SIGKILL(OOM-killer), segv=SIGSEGV(segfault), abort=SIGABRT(CUDA-fatal)")
    ap.add_argument("--tenants", required=True, help="JSON list of {id,url,payload}")
    ap.add_argument("--victim-index", type=int, default=0)
    ap.add_argument("--victim-pid", type=int, required=True)
    ap.add_argument("--fault-delay", type=float, default=3.0)
    ap.add_argument("--req-timeout", type=float, default=90.0)
    ap.add_argument("--recovery-timeout", type=float, default=20.0)
    ap.add_argument("--recovery-url", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tenants = json.loads(args.tenants)
    fr = run(tenants, args.victim_index, args.victim_pid, args.fault_type,
             args.system, args.fault_delay, args.req_timeout, args.recovery_timeout,
             recovery_url=args.recovery_url, fault_signal=args.fault_signal)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fr.to_dict(), f, indent=2)
    d = fr.to_dict()
    print("=== fault-isolation result ===")
    for k in ("system", "fault_type", "num_tenants", "inflight_at_fault",
              "affected_cotenants", "blast_radius", "request_loss_rate",
              "recovery_time_ms", "failed_requests"):
        print(f"  {k} = {d[k]}")
    p = d["params"]
    print(f"  fault_signal = {p['fault_signal']}")
    print(f"  survivors_completed = {p['survivors_completed']}/{p['cotenants_total']} "
          f"(all={p['all_survivors_completed']}, min_tokens={p['min_survivor_tokens']})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
