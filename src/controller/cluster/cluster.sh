#!/usr/bin/env bash
# cluster.sh — Start/stop/restart/status/test the multi-GPU cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$SCRIPT_DIR/cluster.conf"

if [[ ! -f "$CONF" ]]; then
  echo "ERROR: $CONF not found"; exit 1
fi
source "$CONF"

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

PYTHON="$PROJECT_ROOT/venv/bin/python"

# ── Helpers ──────────────────────────────────────────────────────────────────

parse_node() {
  # "node-master:local:cuda:0:24" → NODE_ID  HOST  DEVICE  NODE_MAX_WORKERS
  local IFS=':'
  read -r NODE_ID HOST CUDA_A CUDA_B NODE_MAX_WORKERS <<< "$1"
  DEVICE="${CUDA_A}:${CUDA_B}"
}

# Compute per-node ports based on index (0-based).
# This allows multiple local nodes without port conflicts.
set_node_ports() {
  local idx=$1  # 0-based node index
  CUR_NODE_AGENT_PORT=$((NODE_AGENT_PORT + idx))
  CUR_CONTROLLER_PORT=$((CONTROLLER_PORT + idx))
  CUR_AGG_HEALTH_PORT=$((AGGREGATOR_HEALTH_PORT + idx))
  CUR_AGG_TCP_PORT=$((50056 + idx))
  CUR_WORKER_BASE_PORT=$((5000 + idx * 1000))
}

kill_ports_local() {
  for port in "$@"; do
    fuser -k -TERM "$port/tcp" 2>/dev/null || true
  done
  sleep 3
  for port in "$@"; do
    fuser -k -KILL "$port/tcp" 2>/dev/null || true
  done
}

kill_ports_remote() {
  local host=$1; shift
  local ssh_port
  ssh_port=$(get_ssh_port "$host")
  # Build fuser commands for each port, with lsof fallback if fuser unavailable
  local kill_cmd=""
  for port in "$@"; do
    kill_cmd+="fuser -k -KILL $port/tcp 2>/dev/null || lsof -ti:$port | xargs kill -9 2>/dev/null; "
  done
  ssh -p "$ssh_port" -o ConnectTimeout=5 -o StrictHostKeyChecking=no -o BatchMode=yes "$host" "$kill_cmd" 2>/dev/null || true
}

get_ssh_port() {
  local host=$1
  echo "${SSH_PORTS[$host]:-22}"
}

# Wrapper for SSH to remote nodes with correct port and options
remote_ssh() {
  local host=$1; shift
  local ssh_port
  ssh_port=$(get_ssh_port "$host")
  ssh -p "$ssh_port" -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes "$host" "$@"
}

setup_tunnel() {
  local host=$1
  local local_port=$2
  local ssh_port
  ssh_port=$(get_ssh_port "$host")

  echo "    Setting up SSH tunnel to $host (port $ssh_port) ..."
  echo "      -L $local_port → remote:$NODE_AGENT_PORT (forward: node agent)"
  echo "      -R $GLOBAL_CONTROLLER_PORT → master:$GLOBAL_CONTROLLER_PORT (reverse: global controller)"

  ssh -f -N \
    -p "$ssh_port" \
    -L "${local_port}:localhost:${NODE_AGENT_PORT}" \
    -R "${GLOBAL_CONTROLLER_PORT}:localhost:${GLOBAL_CONTROLLER_PORT}" \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    "$host"

  echo "      Tunnel established."
}

kill_tunnels() {
  local host=$1
  pkill -f "ssh -f -N.*${host}" 2>/dev/null || true
}

# Kill all cluster processes on a given host (local or remote)
kill_all_on_host() {
  local host=$1  # "local" or hostname

  if [[ "$host" == "local" ]]; then
    # Kill all possible node ports (iterate through all NODES entries)
    local all_ports="$GLOBAL_CONTROLLER_PORT"
    local idx=0
    for entry in "${NODES[@]}"; do
      set_node_ports "$idx"
      all_ports="$all_ports $CUR_NODE_AGENT_PORT $CUR_CONTROLLER_PORT $CUR_AGG_HEALTH_PORT"
      idx=$((idx + 1))
    done
    kill_ports_local $all_ports
    pkill -f "worker_sync.py --http-port" 2>/dev/null || true
    pkill -f "template_process.py" 2>/dev/null || true
    pkill -f "aggregator" 2>/dev/null || true
    pkill -f "controller.controller" 2>/dev/null || true
    pkill -f "controller.cluster.node_agent" 2>/dev/null || true
    pkill -f "controller.cluster.global_controller" 2>/dev/null || true
  else
    # Use base ports for remote (each remote node has its own port space)
    kill_ports_remote "$host" "$NODE_AGENT_PORT" "$CONTROLLER_PORT" "$AGGREGATOR_HEALTH_PORT"
    # Bracket trick: pkill -f '[p]attern' matches target processes but NOT the
    # bash shell running the SSH command (whose cmdline contains '[p]attern' literally).
    # Without this, pkill kills the SSH session itself → exit 255 → remaining kills never run.
    remote_ssh "$host" \
      "pkill -9 -f '[w]orker_sync.py --http-port' 2>/dev/null; \
       pkill -9 -f '[t]emplate_process.py' 2>/dev/null; \
       pkill -9 -f '[a]ggregator_sync' 2>/dev/null; \
       pkill -9 -f '[s]erverless.controller' 2>/dev/null; \
       pkill -9 -f '[s]erverless.cluster.node_agent' 2>/dev/null" || true
    kill_tunnels "$host"
  fi
}

# ── start ────────────────────────────────────────────────────────────────────

do_start() {
  echo "=== Starting cluster ==="

  # 0. Clean up any stale processes from a previous run
  echo "[0] Cleaning up stale processes ..."
  local cleanup_idx=0
  for entry in "${NODES[@]}"; do
    parse_node "$entry"
    set_node_ports "$cleanup_idx"
    cleanup_idx=$((cleanup_idx + 1))
    kill_all_on_host "$HOST"
  done
  sleep 1
  echo "    Clean."

  # 0.5. Setup MPS on all nodes (local + remote)
  echo "[0.5] Setting up MPS on all nodes ..."
  # setup_mps.sh lives in scripts/; the guard and the stop call both pointed at
  # $PROJECT_ROOT/setup_mps.sh, so the guard was always false and MPS never
  # started on the master -- while remote nodes got it unconditionally below.
  if [[ -f "$PROJECT_ROOT/scripts/setup_mps.sh" ]]; then
    bash "$PROJECT_ROOT/scripts/setup_mps.sh" stop 2>/dev/null || true
    sleep 1
    bash "$PROJECT_ROOT/scripts/setup_mps.sh" start
    echo "    MPS started on local node."
  fi
  for entry in "${NODES[@]}"; do
    parse_node "$entry"
    if [[ "$HOST" != "local" ]]; then
      local rroot="${REMOTE_PROJECT_ROOT:-$PROJECT_ROOT}"
      echo "    Setting up MPS on $HOST ..."
      remote_ssh "$HOST" \
        "cd $rroot && bash scripts/setup_mps.sh stop 2>/dev/null; sleep 1; bash scripts/setup_mps.sh start" \
        2>/dev/null || echo "    WARNING: MPS setup failed on $HOST"
    fi
  done

  # 1. Global controller
  echo "[1] Starting global controller on port $GLOBAL_CONTROLLER_PORT ..."
  nohup "$PYTHON" -m controller.cluster.global_controller \
    --host 0.0.0.0 \
    --port "$GLOBAL_CONTROLLER_PORT" \
    --max-nodes "$MAX_NODES" \
    --min-nodes "$MIN_NODES" \
    --placement-policy "${PLACEMENT_POLICY:-affinity}" \
    --adapters-per-model "${ADAPTERS_PER_MODEL:-50}" \
    --adapter-prefix "${ADAPTER_PREFIX:-../sim-adapters/pool-10-r16/lora-}" \
    > "$LOG_DIR/global_controller.log" 2>&1 &
  echo "    PID=$!"

  echo "    Waiting for controller to bind ..."
  local gc_ready=false
  for _i in $(seq 1 20); do
    if curl -sf "http://localhost:${GLOBAL_CONTROLLER_PORT}/health" >/dev/null 2>&1; then
      gc_ready=true
      break
    fi
    sleep 0.5
  done
  if [[ "$gc_ready" == "true" ]]; then
    echo "    Global controller ready."
  else
    echo "    ERROR: Global controller failed to start (check $LOG_DIR/global_controller.log)"
    exit 1
  fi

  # 2. Launch node agents (with SSH tunnels for remote nodes)
  local i=0
  local tunnel_port=${TUNNEL_BASE_PORT:-9101}

  for entry in "${NODES[@]}"; do
    parse_node "$entry"
    set_node_ports "$i"
    i=$((i + 1))

    echo "[$((i + 1))] Starting $NODE_ID (host=$HOST, device=$DEVICE, workers=$NODE_MAX_WORKERS) ..."

    # Check if this node uses single-aggregator mode
    local single_agg_flag=""
    if [[ " ${SINGLE_AGG_NODES:-} " == *" $NODE_ID "* ]]; then
      single_agg_flag="--single-aggregator"
    fi

    if [[ "$HOST" == "local" ]]; then
      nohup "$PYTHON" -m controller.cluster.node_agent \
        --node-id "$NODE_ID" \
        --port "$CUR_NODE_AGENT_PORT" \
        --global-controller "http://${MASTER_IP}:${GLOBAL_CONTROLLER_PORT}" \
        --aggregator-device "$DEVICE" \
        --aggregator-health-port "$CUR_AGG_HEALTH_PORT" \
        --aggregator-tcp-port "$CUR_AGG_TCP_PORT" \
        --controller-port "$CUR_CONTROLLER_PORT" \
        --worker-base-port "$CUR_WORKER_BASE_PORT" \
        --max-workers "$NODE_MAX_WORKERS" \
        --scheduler "$SCHEDULER" \
        --scale-down-delay "${SCALE_DOWN_DELAY:-5}" \
        $single_agg_flag \
        > "$LOG_DIR/${NODE_ID}.log" 2>&1 &
      echo "    PID=$!"

    else
      local local_agent_port=$tunnel_port
      tunnel_port=$((tunnel_port + 1))
      local ssh_port
      ssh_port=$(get_ssh_port "$HOST")
      local remote_root="${REMOTE_PROJECT_ROOT:-$PROJECT_ROOT}"
      local remote_python="$remote_root/venv/bin/python"
      local remote_log_dir="$remote_root/logs"

      setup_tunnel "$HOST" "$local_agent_port"

      ssh -p "$ssh_port" -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o BatchMode=yes -f "$HOST" \
        "export LD_LIBRARY_PATH=\$(ls -d $remote_root/venv/lib/python3*/site-packages/torch/lib 2>/dev/null | head -1):\$LD_LIBRARY_PATH; \
         export PYTHONPATH=$remote_root/src:\$PYTHONPATH; \
         export SWARM_EVENT_LOG=${SWARM_EVENT_LOG:-$remote_root/logs/pool_events.jsonl}; \
         export OPENBLAS_NUM_THREADS=1; \
         export OMP_NUM_THREADS=1; \
         export CUDA_MODULE_LOADING=LAZY; \
         export HF_HOME=\${HF_HOME:-/root/.cache/huggingface}; \
         mkdir -p $remote_log_dir && cd $remote_root && nohup $remote_python -m controller.cluster.node_agent \
          --node-id $NODE_ID \
          --port $NODE_AGENT_PORT \
          --global-controller http://localhost:${GLOBAL_CONTROLLER_PORT} \
          --aggregator-device $DEVICE \
          --max-workers $NODE_MAX_WORKERS \
          --scheduler $SCHEDULER \
          --scale-down-delay ${SCALE_DOWN_DELAY:-5} \
          --register-host localhost \
          --register-port $local_agent_port \
          $single_agg_flag \
          > $remote_log_dir/${NODE_ID}.log 2>&1 </dev/null &"
      echo "    Started via SSH (tunnel port $local_agent_port)"
    fi
  done

  # 3. Wait for all nodes to register and become healthy
  echo ""
  echo "Waiting for nodes to register ..."
  local expected=${#NODES[@]}
  local registered=0
  for _i in $(seq 1 30); do
    registered=$(curl -sf "http://localhost:${GLOBAL_CONTROLLER_PORT}/cluster/state" 2>/dev/null \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['summary']['total_nodes'])" 2>/dev/null) || registered=0
    if [[ "$registered" -ge "$expected" ]]; then
      echo "  All $expected nodes registered."
      break
    fi
    sleep 1
  done
  if [[ "$registered" -lt "$expected" ]]; then
    echo "  WARNING: Only $registered/$expected nodes registered after 30s"
  fi

  # 4. Wait for aggregators to launch (auto-scaler needs ~5-10s per node)
  echo "Waiting for aggregators to launch ..."
  local active=0
  local agg_timeout=${AGG_TIMEOUT:-300}
  local agg_iters=$(( agg_timeout / 2 ))
  for _i in $(seq 1 $agg_iters); do
    active=$(curl -sf "http://localhost:${GLOBAL_CONTROLLER_PORT}/cluster/state" 2>/dev/null \
      | python3 -c "import sys,json; d=json.load(sys.stdin)['summary']; print(d['active_nodes'])" 2>/dev/null) || active=0
    if [[ "$active" -ge "$expected" ]]; then
      echo "  All $expected nodes active (aggregators running)."
      break
    fi
    sleep 2
  done
  if [[ "$active" -lt "$expected" ]]; then
    echo "  WARNING: Only $active/$expected nodes active after ${agg_timeout}s (check logs)"
  fi

  echo ""
  do_status
}

# ── stop ─────────────────────────────────────────────────────────────────────

do_stop() {
  echo "=== Stopping cluster ==="

  local stop_idx=0
  for entry in "${NODES[@]}"; do
    parse_node "$entry"
    set_node_ports "$stop_idx"
    stop_idx=$((stop_idx + 1))
    echo "  Killing all processes on $NODE_ID ($HOST) ..."
    kill_all_on_host "$HOST"
  done

  sleep 1

  # Stop MPS on all nodes
  echo "  Stopping MPS on all nodes ..."
  if [[ -f "$PROJECT_ROOT/scripts/setup_mps.sh" ]]; then
    bash "$PROJECT_ROOT/scripts/setup_mps.sh" stop 2>/dev/null || true
  fi
  for entry in "${NODES[@]}"; do
    parse_node "$entry"
    if [[ "$HOST" != "local" ]]; then
      local rroot="${REMOTE_PROJECT_ROOT:-$PROJECT_ROOT}"
      remote_ssh "$HOST" \
        "cd $rroot && bash scripts/setup_mps.sh stop 2>/dev/null" || true
    fi
  done

  # Clean logs from this run (opened in append mode, avoids stale/corrupted data)
  local log_dir="$PROJECT_ROOT/logs"
  if [ -d "$log_dir" ]; then
    rm -f "$log_dir"/controller*.log "$log_dir"/aggregator*.log "$log_dir"/worker_*.log "$log_dir"/template.log "$log_dir"/template*.sock
    echo "  Logs cleaned."
  fi

  echo "  Done."
}

# ── restart ──────────────────────────────────────────────────────────────────

do_restart() {
  do_stop
  echo ""
  sleep 1
  do_start
}

# ── status ───────────────────────────────────────────────────────────────────

do_status() {
  echo "=== Cluster Status ==="
  echo ""

  # Global controller
  echo "-- Global Controller (port $GLOBAL_CONTROLLER_PORT) --"
  local gc_resp
  if gc_resp=$(curl -sf "http://localhost:${GLOBAL_CONTROLLER_PORT}/cluster/state" 2>/dev/null); then
    # Summary line
    local total active idle
    total=$(echo "$gc_resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['summary']['total_nodes'])" 2>/dev/null) || total="?"
    active=$(echo "$gc_resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['summary']['active_nodes'])" 2>/dev/null) || active="?"
    idle=$(echo "$gc_resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['summary']['idle_nodes'])" 2>/dev/null) || idle="?"
    echo "  Nodes: $total total, $active active, $idle idle"
    echo ""

    # Per-node details from global controller
    echo "$gc_resp" | python3 -c "
import sys, json
data = json.load(sys.stdin)
for nid, info in data.get('nodes', {}).items():
    status = info.get('status', '?')
    gpu = info.get('gpu_type', '?')
    gpu_mem = info.get('gpu_memory_gb', 0)
    workers = info.get('active_workers', 0)
    max_w = info.get('max_workers', 0)
    adapters = info.get('loaded_adapters', [])
    host = info.get('host', '?')
    port = info.get('port', '?')
    print(f'  {nid}: status={status}, workers={workers}/{max_w}, gpu={gpu} ({gpu_mem:.0f}GB), addr={host}:{port}')
    if adapters:
        print(f'    adapters: {adapters}')
" 2>/dev/null || echo "$gc_resp" | python3 -m json.tool 2>/dev/null || echo "$gc_resp"
  else
    echo "  NOT RUNNING"
  fi
  echo ""

  # Per-node process info
  local tunnel_port=${TUNNEL_BASE_PORT:-9101}
  local node_idx=0
  for entry in "${NODES[@]}"; do
    parse_node "$entry"
    set_node_ports "$node_idx"
    node_idx=$((node_idx + 1))
    echo "-- $NODE_ID ($HOST, $DEVICE) --"

    if [[ "$HOST" == "local" ]]; then
      # Node agent
      local pid
      pid=$(fuser "$CUR_NODE_AGENT_PORT/tcp" 2>/dev/null | xargs) || true
      if [[ -n "$pid" ]]; then
        echo "  node_agent: PID $pid (port $CUR_NODE_AGENT_PORT)"
      else
        echo "  node_agent: NOT RUNNING"
      fi

      # Controller
      local ctrl_pid
      ctrl_pid=$(fuser "$CUR_CONTROLLER_PORT/tcp" 2>/dev/null | xargs) || true
      if [[ -n "$ctrl_pid" ]]; then
        echo "  controller: PID $ctrl_pid (port $CUR_CONTROLLER_PORT)"
      else
        echo "  controller: NOT RUNNING"
      fi

      # Aggregator
      local agg_pid
      agg_pid=$(fuser "$CUR_AGG_HEALTH_PORT/tcp" 2>/dev/null | xargs) || true
      if [[ -n "$agg_pid" ]]; then
        echo "  aggregator: PID $agg_pid (port $CUR_AGG_HEALTH_PORT)"
      else
        echo "  aggregator: NOT RUNNING"
      fi

      # Workers (count via node agent API for this specific node)
      local worker_count
      worker_count=$(curl -sf "http://localhost:${CUR_NODE_AGENT_PORT}/status" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('active_workers',0))" 2>/dev/null) || worker_count=0
      echo "  workers: $worker_count running"

      # Health check
      local health
      if health=$(curl -sf "http://localhost:${CUR_NODE_AGENT_PORT}/health" 2>/dev/null); then
        echo "  health: ok"
      else
        echo "  health: UNREACHABLE"
      fi

    else
      # SSH tunnel
      local tunnel_pid
      tunnel_pid=$(pgrep -f "ssh -f -N.*${HOST}" 2>/dev/null | head -1) || true
      if [[ -n "$tunnel_pid" ]]; then
        echo "  ssh_tunnel: PID $tunnel_pid (local port $tunnel_port)"
      else
        echo "  ssh_tunnel: NOT RUNNING"
      fi

      # Remote node agent
      local remote_pid
      remote_pid=$(remote_ssh "$HOST" \
        "fuser $CUR_NODE_AGENT_PORT/tcp 2>/dev/null | xargs" 2>/dev/null) || true
      if [[ -n "$remote_pid" ]]; then
        echo "  node_agent: PID $remote_pid (remote, port $CUR_NODE_AGENT_PORT)"
      else
        echo "  node_agent: NOT RUNNING"
      fi

      # Remote controller
      local remote_ctrl_pid
      remote_ctrl_pid=$(remote_ssh "$HOST" \
        "fuser $CUR_CONTROLLER_PORT/tcp 2>/dev/null | xargs" 2>/dev/null) || true
      if [[ -n "$remote_ctrl_pid" ]]; then
        echo "  controller: PID $remote_ctrl_pid (remote, port $CUR_CONTROLLER_PORT)"
      else
        echo "  controller: NOT RUNNING"
      fi

      # Remote aggregator
      local remote_agg_pid
      remote_agg_pid=$(remote_ssh "$HOST" \
        "fuser $CUR_AGG_HEALTH_PORT/tcp 2>/dev/null | xargs" 2>/dev/null) || true
      if [[ -n "$remote_agg_pid" ]]; then
        echo "  aggregator: PID $remote_agg_pid (remote, port $CUR_AGG_HEALTH_PORT)"
      else
        echo "  aggregator: NOT RUNNING"
      fi

      # Remote workers
      local remote_worker_count
      remote_worker_count=$(remote_ssh "$HOST" \
        "pgrep -fc 'worker_sync.py --http-port' 2>/dev/null || echo 0" 2>/dev/null) || remote_worker_count=0
      echo "  workers: $remote_worker_count running (remote)"

      # Health check via tunnel
      local health
      if health=$(curl -sf "http://localhost:${tunnel_port}/health" 2>/dev/null); then
        echo "  health: ok (via tunnel)"
      else
        echo "  health: UNREACHABLE (via tunnel)"
      fi

      tunnel_port=$((tunnel_port + 1))
    fi
  done
}

# ── test ─────────────────────────────────────────────────────────────────────

do_test() {
  echo "=== Test Request ==="
  curl -s "http://localhost:${GLOBAL_CONTROLLER_PORT}/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"${DEFAULT_ADAPTER}\",
      \"prompt\": \"The meaning of life is\",
      \"max_tokens\": 64,
      \"temperature\": 0.7
    }" | python3 -m json.tool 2>/dev/null || echo "(request failed)"
}

# ── stress ───────────────────────────────────────────────────────────────────

do_stress() {
  local n=${1:-8}
  echo "=== Stress Test: $n concurrent requests ==="

  local tmpdir
  tmpdir=$(mktemp -d)
  local pids=()

  for i in $(seq 1 "$n"); do
    (
      local start_ms=$(($(date +%s%N) / 1000000))
      local http_code
      http_code=$(curl -s -o "$tmpdir/resp_$i.json" -w "%{http_code}" \
        "http://localhost:${GLOBAL_CONTROLLER_PORT}/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{
          \"model\": \"${DEFAULT_ADAPTER}\",
          \"prompt\": \"Write a short sentence about topic $i:\",
          \"max_tokens\": 32,
          \"temperature\": 0.8
        }")
      local end_ms=$(($(date +%s%N) / 1000000))
      local elapsed=$(( end_ms - start_ms ))
      echo "$i $http_code $elapsed" > "$tmpdir/stat_$i.txt"
    ) &
    pids+=($!)
  done

  echo "  Waiting for $n requests ..."
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  # Summary
  echo ""
  printf "  %-5s %-6s %-10s\n" "REQ" "HTTP" "TIME(ms)"
  printf "  %-5s %-6s %-10s\n" "---" "----" "--------"

  local total_ms=0 ok=0 fail=0
  for i in $(seq 1 "$n"); do
    if [[ -f "$tmpdir/stat_$i.txt" ]]; then
      read -r req code ms < "$tmpdir/stat_$i.txt"
      printf "  %-5s %-6s %-10s\n" "$req" "$code" "$ms"
      total_ms=$((total_ms + ms))
      if [[ "$code" == "200" ]]; then ok=$((ok+1)); else fail=$((fail+1)); fi
    else
      printf "  %-5s %-6s %-10s\n" "$i" "ERR" "-"
      fail=$((fail+1))
    fi
  done

  echo ""
  echo "  OK: $ok  FAIL: $fail  AVG: $((total_ms / n))ms"

  rm -rf "$tmpdir"
}

# ── prewarm ──────────────────────────────────────────────────────────────────

do_prewarm() {
  local trace_file="${1:?Usage: cluster.sh prewarm <trace.jsonl> [adapter-prefix]}"
  local adapter_prefix="${2:-../sim-adapters/pool-10-r16/lora-}"

  if [[ ! -f "$trace_file" ]]; then
    echo "ERROR: trace file not found: $trace_file"; exit 1
  fi

  echo "=== Pre-warming workers ==="
  echo "  Trace: $trace_file"
  echo "  Prefix: $adapter_prefix"

  # Build node list: "host:port:max_workers" for each node
  local node_specs=()
  local tunnel_port=${TUNNEL_BASE_PORT:-9101}
  local pw_idx=0
  for entry in "${NODES[@]}"; do
    parse_node "$entry"
    set_node_ports "$pw_idx"
    pw_idx=$((pw_idx + 1))
    if [[ "$HOST" == "local" ]]; then
      node_specs+=("localhost:${CUR_NODE_AGENT_PORT}:${NODE_MAX_WORKERS}")
    else
      node_specs+=("localhost:${tunnel_port}:${NODE_MAX_WORKERS}")
      tunnel_port=$((tunnel_port + 1))
    fi
  done

  # Build device specs
  local device_specs=""
  if [[ -n "${PREWARM_DEVICES+x}" ]]; then
    for dentry in "${PREWARM_DEVICES[@]}"; do
      local node_id="${dentry%%:*}"
      local rest="${dentry#*:}"
      local find_tunnel=${TUNNEL_BASE_PORT:-9101}
      local find_idx=0
      for entry in "${NODES[@]}"; do
        parse_node "$entry"
        set_node_ports "$find_idx"
        find_idx=$((find_idx + 1))
        if [[ "$NODE_ID" == "$node_id" ]]; then
          if [[ "$HOST" == "local" ]]; then
            device_specs+="localhost:${CUR_NODE_AGENT_PORT}=${rest} "
          else
            device_specs+="localhost:${find_tunnel}=${rest} "
          fi
          break
        fi
        if [[ "$HOST" != "local" ]]; then
          find_tunnel=$((find_tunnel + 1))
        fi
      done
    done
  fi

  "$PYTHON" - "$trace_file" "$adapter_prefix" "$device_specs" "${node_specs[@]}" <<'PYEOF'
import json, sys, collections, math, urllib.request

trace_file = sys.argv[1]
adapter_prefix = sys.argv[2]
device_specs_raw = sys.argv[3].strip()
node_specs_raw = sys.argv[4:]

# Parse device specs: "localhost:9100=cuda:0:12,cuda:1:18 host2:9100=cuda:0:3,cuda:1:10"
node_devices = {}
if device_specs_raw:
    for spec in device_specs_raw.split():
        key, devs = spec.split("=", 1)
        devices = {}
        parts = devs.split(":")
        i = 0
        while i + 2 < len(parts):
            dev = f"{parts[i]}:{parts[i+1]}"
            cnt = int(parts[i+2])
            devices[dev] = cnt
            i += 3
        node_devices[key] = devices

# Parse node specs
nodes = []
for spec in node_specs_raw:
    host, port, max_w = spec.rsplit(":", 2)
    key = f"{host}:{port}"
    nodes.append({"host": host, "port": int(port), "max": int(max_w), "devices": node_devices.get(key, {})})

total_capacity = sum(n["max"] for n in nodes)
print(f"  Total capacity: {total_capacity} workers across {len(nodes)} nodes")
for node in nodes:
    if node["devices"]:
        dev_str = ", ".join(f"{d}={c}" for d, c in node["devices"].items())
        print(f"    {node['host']}:{node['port']}: max={node['max']} ({dev_str})")
    else:
        print(f"    {node['host']}:{node['port']}: max={node['max']} (round-robin)")

# Count adapter popularity from trace
adapter_counts = collections.Counter()
with open(trace_file) as f:
    for line in f:
        r = json.loads(line)
        aid = r.get("adapter_id", r.get("model", ""))
        adapter_counts[aid] += 1

total_requests = sum(adapter_counts.values())
print(f"  Trace: {total_requests} requests, {len(adapter_counts)} adapters")

# Allocate workers proportionally (at least 1 per adapter)
adapter_workers = {}
remaining = total_capacity
sorted_adapters = adapter_counts.most_common()
for i, (aid, count) in enumerate(sorted_adapters):
    mapped = f"{adapter_prefix}{aid}"
    if i == len(sorted_adapters) - 1:
        w = remaining
    else:
        w = max(1, round(total_capacity * count / total_requests))
        w = min(w, remaining)
    adapter_workers[mapped] = w
    remaining -= w
    if remaining <= 0:
        break

print(f"  Allocation ({len(adapter_workers)} adapters):")
for a, w in adapter_workers.items():
    print(f"    {a}: {w} workers")
print(f"  Total workers to spawn: {sum(adapter_workers.values())}")

# Distribute adapters across nodes proportionally
for node in nodes:
    node["adapters"] = {}

for adapter, total_w in adapter_workers.items():
    assigned = 0
    for node in nodes:
        share = max(1, round(total_w * node["max"] / total_capacity))
        cap = node["max"] - sum(node["adapters"].values())
        share = min(share, total_w - assigned, cap)
        if share > 0:
            node["adapters"][adapter] = share
            assigned += share
        if assigned >= total_w:
            break
    while assigned < total_w:
        placed = False
        for node in nodes:
            cap = node["max"] - sum(node["adapters"].values())
            if cap > 0:
                node["adapters"][adapter] = node["adapters"].get(adapter, 0) + 1
                assigned += 1
                placed = True
                break
            if not placed:
                break

# Send prewarm requests concurrently
import concurrent.futures

def prewarm_node(node):
    if not node["adapters"]:
        return None
    url = f"http://{node['host']}:{node['port']}/prewarm"
    body = {"adapters": node["adapters"]}
    if node.get("devices"):
        body["devices"] = node["devices"]
    payload = json.dumps(body).encode()
    total_node_workers = sum(node["adapters"].values())
    dev_str = ", ".join(f"{d}={c}" for d, c in node["devices"].items()) if node.get("devices") else "round-robin"
    print(f"  Prewarming {node['host']}:{node['port']} — {total_node_workers} workers ({dev_str})")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    print(f"    {node['host']}: {result.get('spawned', '?')}/{result.get('total_requested', '?')} spawned")
    return result

with concurrent.futures.ThreadPoolExecutor(max_workers=len(nodes)) as pool:
    futures = {pool.submit(prewarm_node, n): n for n in nodes}
    for fut in concurrent.futures.as_completed(futures):
        node = futures[fut]
        try:
            fut.result()
        except Exception as e:
            print(f"    FAILED on {node['host']}: {e}")
            sys.exit(1)

print("  All prewarm requests sent successfully.")
PYEOF

  echo ""
  echo "  Waiting for workers to initialize ..."
  sleep 15
  do_status
  echo ""
  echo "=== Prewarm complete. Ready for benchmark. ==="
}

# ── main ─────────────────────────────────────────────────────────────────────

case "${1:-}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_restart ;;
  status)  do_status ;;
  test)    do_test ;;
  stress)  do_stress "${2:-8}" ;;
  prewarm) do_prewarm "${2:-}" "${3:-}" ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|test|stress [N]|prewarm <trace> [prefix]}"
    exit 1
    ;;
esac
