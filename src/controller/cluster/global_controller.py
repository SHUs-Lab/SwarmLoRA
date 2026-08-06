#  Global Controller - Cluster-level orchestration for split-model serverless  #
#  Manages: Node scaling (Tier 2), Request routing, Adapter placement          #

import asyncio
import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import aiohttp
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

try:
    from .registry import ClusterRegistry, NodeStatus, NodeInfo
    from .model_config import ClusterModelConfig, create_benchmark_config
    from .placement import create_placement_policy
    from .resilience import RetryExecutor, RetryConfig, CircuitBreaker
except ImportError:
    from registry import ClusterRegistry, NodeStatus, NodeInfo
    from model_config import ClusterModelConfig, create_benchmark_config
    from placement import create_placement_policy
    from resilience import RetryExecutor, RetryConfig, CircuitBreaker

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from config import GLOBAL_CONTROLLER_PORT

try:
    from ..scheduling.base import ClusterScaler, ClusterSnapshot, ScalingAction
    from .default_scaler import DefaultClusterScaler
except ImportError:
    try:
        from controller.scheduling.base import ClusterScaler, ClusterSnapshot, ScalingAction
        from controller.cluster.default_scaler import DefaultClusterScaler
    except ImportError:
        from scheduling.base import ClusterScaler, ClusterSnapshot, ScalingAction
        from cluster.default_scaler import DefaultClusterScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration

@dataclass
class GlobalControllerConfig:
    # API
    host: str = "0.0.0.0"
    port: int = GLOBAL_CONTROLLER_PORT

    # Multi-model configuration
    model_config_path: Optional[str] = None      # Path to model config JSON/YAML
    num_base_models: int = 1                     # For auto-generated benchmark config
    adapters_per_model: int = 50                 # For auto-generated benchmark config
    adapter_prefix: str = "../sim-adapters/pool-10-r16/lora-"  # For auto-generated benchmark config

    # Tier 2 Scaling (Aggregators/Nodes)
    min_active_nodes: int = 0                    # Scale to zero enabled if 0
    max_active_nodes: int = 10                   # Max nodes with aggregators
    node_scale_up_threshold: int = 1             # Pending requests to trigger scale up
    node_scale_down_delay: float = 300.0         # 5 min idle before scale down
    node_startup_timeout: float = 120.0          # Max time to wait for node ready

    # Health & Monitoring
    heartbeat_timeout: float = 30.0              # Mark node failed if no heartbeat
    health_check_interval: float = 10.0          # How often to check node health
    metrics_interval: float = 30.0               # How often to log metrics

    # Request handling
    request_timeout: float = 300.0               # Max time for a request
    max_queue_size: int = 1000                   # Max pending requests

    # Placement
    placement_policy: str = "affinity"           # Placement policy name (see placement.py)

    # Cluster scaler (pluggable auto-scaling policy)
    scaler: Optional[ClusterScaler] = None       # None = DefaultClusterScaler

    # Resilience
    retry_max_attempts: int = 3                  # Max retry attempts for node forwarding
    retry_base_delay: float = 0.5                # Base delay between retries (seconds)
    forward_max_attempts: int = 2                # Max node-forward attempts per request

    # Latency tracking
    latency_buffer_max: int = 1000               # Global latency ring buffer size
    per_node_latency_buffer_max: int = 200       # Per-node latency ring buffer size

    # Auto-scaler
    auto_scaler_interval: float = 5.0            # How often to run auto-scaler (seconds)

    # Scale-up wait
    scale_up_wait_timeout: float = 60.0          # Max seconds to wait for a node to scale up
    scale_up_poll_interval: float = 0.2          # Poll interval during scale-up wait

    # Scale-down
    graceful_drain_timeout: float = 30.0         # Graceful drain timeout for node shutdown

    # Idempotency
    idempotency_cleanup_interval: float = 300.0  # How often to clean expired entries


# Request/Response Models

class NodeRegistration(BaseModel):
    node_id: str
    host: str
    port: int
    gpu_type: str = "unknown"
    gpu_memory_gb: float = 0
    num_gpus: int = 0
    max_workers: int = 32
    base_model: Optional[str] = None      # Which base model this node will run


class NodeHeartbeat(BaseModel):
    node_id: str
    status: str
    aggregator_pid: Optional[int] = None
    active_workers: int = 0
    total_workers: int = 0                # Actually spawned and ready
    free_workers: int = 0                 # Ready but idle
    queue_depth: int = 0                  # Requests queued at local controller
    max_workers: Optional[int] = None     # Effective max (may be GPU-capped)
    num_gpus: Optional[int] = None        # Number of GPUs (dynamic refresh)
    gpu_type: Optional[str] = None        # GPU type (dynamic refresh)
    loaded_adapters: List[str] = []
    base_model: Optional[str] = None      # Current base model
    node_metrics: Optional[Dict] = None   # Aggregate load metrics from local controller


class InferenceRequest(BaseModel):
    model: Optional[str] = None         # adapter name (OpenAI-compatible)
    adapter_id: Optional[str] = None    # adapter name (benchmark-compatible)
    prompt: Optional[str] = None
    messages: Optional[List[dict]] = None
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False

    @property
    def adapter_name(self) -> str:
        """Resolve adapter name from either field."""
        return self.model or self.adapter_id or ""


class ClusterMetrics(BaseModel):
    total_nodes: int
    active_nodes: int
    idle_nodes: int
    total_workers: int
    pending_requests: int
    total_requests: int
    requests_per_second: float


# Global Controller

class GlobalController:
    """Global controller for the serverless LLM cluster."""

    def __init__(self, config: GlobalControllerConfig = None, model_config: ClusterModelConfig = None):
        self.config = config or GlobalControllerConfig()
        self.registry = ClusterRegistry()

        # Load or create model configuration
        if model_config:
            self.model_config = model_config
        elif self.config.model_config_path:
            if self.config.model_config_path.endswith('.yaml'):
                self.model_config = ClusterModelConfig.from_yaml(self.config.model_config_path)
            else:
                self.model_config = ClusterModelConfig.from_json(self.config.model_config_path)
            logger.info(f"Loaded model config from {self.config.model_config_path}")
        else:
            # Create default benchmark config
            self.model_config = create_benchmark_config(
                num_base_models=self.config.num_base_models,
                adapters_per_model=self.config.adapters_per_model,
                adapter_prefix=self.config.adapter_prefix,
            )
            logger.info(f"Created benchmark config: {self.config.num_base_models} base models, "
                       f"{self.config.adapters_per_model} adapters each, prefix={self.config.adapter_prefix}")

        self.placement = create_placement_policy(
            policy_name=self.config.placement_policy,
            registry=self.registry,
            model_config=self.model_config,
        )

        # Cluster auto-scaler (pluggable)
        self.scaler = self.config.scaler or DefaultClusterScaler(
            scale_up_threshold=self.config.node_scale_up_threshold,
        )

        # Resilience
        self._node_circuits: Dict[str, CircuitBreaker] = {}
        self._retry_config = RetryConfig(max_attempts=self.config.retry_max_attempts, base_delay=self.config.retry_base_delay)
        self._retry = RetryExecutor(self._retry_config)

        # Idempotency cache: key -> (timestamp, result)
        self._idempotency_cache: Dict[str, tuple] = {}
        self._idempotency_ttl = 3600.0
        self._idempotency_lock = asyncio.Lock()

        # Request tracking
        self._pending_requests: int = 0
        self._inflight_per_node: Dict[str, int] = {}  # node_id -> in-flight count

        # Metrics
        self._total_requests: int = 0
        self._completed_requests: int = 0
        self._failed_requests: int = 0
        self._requests_last_minute: List[float] = []

        # Latency tracking (ring buffer of recent latencies in ms)
        self._latency_buffer: List[float] = []
        self._latency_buffer_max = self.config.latency_buffer_max
        self._per_node_latency: Dict[str, List[float]] = {}  # node_id -> ring buffer
        self._per_node_latency_max = self.config.per_node_latency_buffer_max

        # Per-node error/success counters
        self._per_node_completed: Dict[str, int] = {}
        self._per_node_errors: Dict[str, int] = {}

        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._session: Optional[aiohttp.ClientSession] = None
        self._scale_up_lock = asyncio.Lock()

        # API
        self.app = FastAPI(title="Global Controller")
        self._setup_routes()

    def _setup_routes(self):
        app = self.app

        @app.on_event("startup")
        async def startup():
            await self.start()

        @app.on_event("shutdown")
        async def shutdown():
            await self.stop()

        # =====================================================================
        # Node Management (called by node agents)
        # =====================================================================

        @app.post("/nodes/register")
        async def register_node(req: NodeRegistration):
            node = await self.registry.register_node(
                node_id=req.node_id,
                host=req.host,
                port=req.port,
                gpu_type=req.gpu_type,
                gpu_memory_gb=req.gpu_memory_gb,
                num_gpus=req.num_gpus,
                max_workers=req.max_workers,
            )
            # Set base model if provided
            if req.base_model:
                await self.registry.set_node_base_model(req.node_id, req.base_model)
                # Track in model_config
                self.model_config.assign_node_to_base_model(req.node_id, req.base_model)

            return {"status": "registered", "node_id": node.node_id, "base_model": req.base_model}

        @app.post("/nodes/unregister")
        async def unregister_node(req: dict):
            node = await self.registry.get_node(req["node_id"])
            if node and node.base_model:
                self.model_config.remove_node_from_base_model(req["node_id"], node.base_model)
            await self.registry.unregister_node(req["node_id"])
            return {"status": "unregistered"}

        @app.post("/nodes/heartbeat")
        async def node_heartbeat(req: NodeHeartbeat):
            # Update node status in registry
            status = NodeStatus(req.status) if req.status in [s.value for s in NodeStatus] else NodeStatus.ACTIVE
            await self.registry.update_node_status(req.node_id, status)
            await self.registry.update_node_heartbeat(
                node_id=req.node_id,
                active_workers=req.active_workers,
                total_workers=getattr(req, 'total_workers', 0),
                free_workers=getattr(req, 'free_workers', 0),
                queue_depth=getattr(req, 'queue_depth', 0),
                max_workers=req.max_workers,
                num_gpus=req.num_gpus,
                gpu_type=req.gpu_type,
                loaded_adapters=req.loaded_adapters,
                aggregator_pid=req.aggregator_pid,
                node_metrics=req.node_metrics,
            )
            # Update base model if changed
            if req.base_model:
                await self.registry.set_node_base_model(req.node_id, req.base_model)
            return {"status": "ok"}

        # =====================================================================
        # Inference API (called by clients)
        # =====================================================================

        @app.post("/v1/completions")
        async def completions(req: InferenceRequest, request: Request):
            idem_key = request.headers.get("X-Idempotency-Key")
            return await self._handle_inference(req, endpoint="/v1/completions", idempotency_key=idem_key)

        @app.post("/v1/chat/completions")
        async def chat_completions(req: InferenceRequest, request: Request):
            idem_key = request.headers.get("X-Idempotency-Key")
            return await self._handle_inference(req, endpoint="/v1/chat/completions", idempotency_key=idem_key)

        # =====================================================================
        # Management API
        # =====================================================================

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/cluster/state")
        async def cluster_state():
            return await self.registry.get_cluster_state()

        @app.get("/cluster/metrics")
        async def cluster_metrics():
            return await self._get_metrics()

        @app.get("/v1/models")
        async def list_models():
            """List all adapters (models) with their base model info."""
            all_adapters = self.model_config.get_all_adapters() if self.model_config else []
            loaded_adapters = await self.registry.list_adapters()
            loaded_set = {a.adapter_name for a in loaded_adapters}

            return {
                "object": "list",
                "data": [
                    {
                        "id": adapter,
                        "object": "model",
                        "base_model": self.model_config.get_base_model_for_adapter(adapter) if self.model_config else None,
                        "loaded": adapter in loaded_set,
                    }
                    for adapter in all_adapters
                ]
            }

        @app.get("/v1/base_models")
        async def list_base_models():
            """List all base models and their stats."""
            if not self.model_config:
                return {"base_models": []}

            stats = await self.registry.get_base_model_stats()

            return {
                "object": "list",
                "data": [
                    {
                        "name": name,
                        "model_id": cfg.model_id,
                        "num_adapters": len(cfg.adapters),
                        "active_nodes": stats.get(name, {}).get("active_nodes", 0),
                        "total_workers": stats.get(name, {}).get("total_workers", 0),
                        "total_requests": stats.get(name, {}).get("total_requests", 0),
                    }
                    for name, cfg in self.model_config.base_models.items()
                ]
            }

        @app.post("/cluster/scale")
        async def manual_scale(req: dict):
            """Manual scale up/down for testing"""
            action = req.get("action")
            if action == "up":
                node = await self._scale_up_node()
                return {"status": "scaled_up", "node": node.node_id if node else None}
            elif action == "down":
                node_id = req.get("node_id")
                if node_id:
                    await self._scale_down_node(node_id)
                return {"status": "scaled_down"}
            return {"status": "invalid_action"}

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self):
        """Start the global controller"""
        logger.info("Starting global controller")
        self._running = True
        self._session = aiohttp.ClientSession()

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._health_monitor_loop()),
            asyncio.create_task(self._auto_scaler_loop()),
            asyncio.create_task(self._metrics_loop()),
            asyncio.create_task(self._idempotency_cleanup_loop()),
        ]

        logger.info(f"Global controller started on port {self.config.port}")

    async def stop(self):
        """Stop the global controller"""
        logger.info("Stopping global controller")
        self._running = False

        for task in self._tasks:
            task.cancel()
        # Wait for tasks to finish cancellation
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        if self._session:
            await self._session.close()

        logger.info("Global controller stopped")

    # =========================================================================
    # Request Handling
    # =========================================================================

    def _record_latency(self, latency_ms: float, node_id: str = None):
        """Record a request latency measurement (global + per-node)."""
        self._latency_buffer.append(latency_ms)
        if len(self._latency_buffer) > self._latency_buffer_max:
            self._latency_buffer = self._latency_buffer[-self._latency_buffer_max:]

        if node_id:
            buf = self._per_node_latency.get(node_id)
            if buf is None:
                buf = []
                self._per_node_latency[node_id] = buf
            buf.append(latency_ms)
            if len(buf) > self._per_node_latency_max:
                self._per_node_latency[node_id] = buf[-self._per_node_latency_max:]

    @staticmethod
    def _percentiles_from(buffer: List[float]) -> dict:
        """Compute p50/p90/p99 from a latency buffer."""
        if not buffer:
            return {"p50": 0, "p90": 0, "p99": 0, "count": 0}
        s = sorted(buffer)
        n = len(s)
        return {
            "p50": s[n // 2],
            "p90": s[int(n * 0.9)],
            "p99": s[int(n * 0.99)],
            "count": n,
        }

    def _compute_percentiles(self) -> dict:
        """Compute p50/p90/p99 from global latency buffer."""
        return self._percentiles_from(self._latency_buffer)

    async def _handle_inference(self, req: InferenceRequest, endpoint: str, idempotency_key: str = None) -> dict:
        """Handle an inference request with optional idempotency support."""
        adapter_name = req.adapter_name

        if idempotency_key:
            async with self._idempotency_lock:
                if idempotency_key in self._idempotency_cache:
                    ts, cached_result = self._idempotency_cache[idempotency_key]
                    if time.time() - ts < self._idempotency_ttl:
                        return {**cached_result, "_cached": True}

        if self._pending_requests >= self.config.max_queue_size:
            raise HTTPException(status_code=503, detail="Request queue full")

        self._total_requests += 1
        self._pending_requests += 1
        self._requests_last_minute.append(time.time())
        arrival_time = time.time()  # Wall clock for cross-node TTFT

        try:
            # Try up to 2 nodes if first fails due to circuit breaker
            tried_nodes: set = set()
            last_error = None

            for attempt in range(self.config.forward_max_attempts):
                # Find or provision a node (excluding already tried nodes)
                node = await self._get_node_for_request(adapter_name, exclude_nodes=tried_nodes)

                if not node:
                    if attempt == 0:
                        self._failed_requests += 1
                        raise HTTPException(status_code=503, detail="No available nodes")
                    break

                tried_nodes.add(node.node_id)
                self._inflight_per_node[node.node_id] = self._inflight_per_node.get(node.node_id, 0) + 1

                await self.registry.record_request(node.node_id, adapter_name)

                try:
                    t0 = time.time()
                    result = await self._forward_to_node(node, endpoint, req, arrival_time)
                    latency_ms = (time.time() - t0) * 1000
                    self._record_latency(latency_ms, node.node_id)
                    self._per_node_completed[node.node_id] = self._per_node_completed.get(node.node_id, 0) + 1
                    self._completed_requests += 1

                    if idempotency_key:
                        async with self._idempotency_lock:
                            self._idempotency_cache[idempotency_key] = (time.time(), result)

                    return result
                except Exception as e:
                    self._per_node_errors[node.node_id] = self._per_node_errors.get(node.node_id, 0) + 1
                    last_error = e
                    # Retry on another node for first attempt (any error)
                    if attempt == 0:
                        logger.warning(f"Node {node.node_id} error: {str(e)[:100]}, trying another")
                        continue
                    # Second attempt failed - give up
                    self._failed_requests += 1
                    raise HTTPException(status_code=502, detail=str(e))

            # If we exhausted retries
            self._failed_requests += 1
            raise HTTPException(status_code=502, detail=str(last_error) if last_error else "No available nodes")
        finally:
            self._pending_requests -= 1
            for nid in tried_nodes:
                self._inflight_per_node[nid] = max(0, self._inflight_per_node.get(nid, 1) - 1)

    async def _get_node_for_request(self, adapter_name: str, exclude_nodes: set = None) -> Optional[NodeInfo]:
        """Get a node to handle the request, scaling up if needed (multi-model aware)"""
        exclude_nodes = exclude_nodes or set()

        base_model = self.placement._get_base_model(adapter_name)

        if base_model:
            logger.debug(f"Adapter {adapter_name} → base model {base_model}")
        else:
            logger.warning(f"Unknown adapter {adapter_name}, no base model mapping")

        node = await self.placement.select_node_for_request(adapter_name, inflight=self._inflight_per_node)
        if node and node.node_id not in exclude_nodes:
            return node

        # No available node - try to scale up
        # Use lock only for the actual scale-up decision, not for queries
        should_scale_up = False
        async with self._scale_up_lock:
            # Check again (another request might have scaled up)
            node = await self.placement.select_node_for_request(adapter_name, inflight=self._inflight_per_node)
            if node and node.node_id not in exclude_nodes:
                return node

            # Check if we can scale up
            active_nodes = await self.registry.get_active_nodes()
            if len(active_nodes) < self.config.max_active_nodes:
                should_scale_up = True

        # Do the actual scale-up outside the lock to avoid blocking other requests
        if should_scale_up:
            node = await self._scale_up_node(base_model=base_model)
            if node and node.node_id not in exclude_nodes:
                return node

        # Still no node - wait for a node to finish scaling up
        # Scale-up involves launching aggregator + loading model, which can take 30-60s
        wait_iters = int(self.config.scale_up_wait_timeout / self.config.scale_up_poll_interval)
        for _ in range(wait_iters):
            await asyncio.sleep(self.config.scale_up_poll_interval)
            node = await self.placement.select_node_for_request(adapter_name, inflight=self._inflight_per_node)
            if node and node.node_id not in exclude_nodes:
                return node

        return None

    def _get_circuit(self, node_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for a node."""
        if node_id not in self._node_circuits:
            self._node_circuits[node_id] = CircuitBreaker(f"node:{node_id}")
        return self._node_circuits[node_id]

    async def _forward_to_node(
        self,
        node: NodeInfo,
        endpoint: str,
        req: InferenceRequest,
        arrival_time: float = None,
    ) -> dict:
        """Forward request to a node's local controller with circuit breaker and retry."""
        circuit = self._get_circuit(node.node_id)

        if not await circuit.acquire():
            raise Exception(f"Node {node.node_id} circuit OPEN")

        if hasattr(req, 'model_dump'):
            payload = req.model_dump(exclude_none=True)
        else:
            payload = req.dict(exclude_none=True)

        if arrival_time:
            payload["arrival_time"] = arrival_time

        async def do_forward():
            async with self._session.post(
                f"{node.url}/forward",
                json={"endpoint": endpoint, "payload": payload},
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            ) as response:
                if response.status == 200:
                    try:
                        result = await response.json()
                    except Exception as e:
                        raise Exception(f"Invalid JSON from node {node.node_id}: {e}")
                    result["_routing"] = {
                        "node_id": node.node_id,
                        "node_host": node.host,
                        "base_model": node.base_model,
                    }
                    return result
                else:
                    text = await response.text()
                    raise Exception(f"Node returned {response.status}: {text[:200]}")

        try:
            result = await self._retry.execute(do_forward)
            await circuit.record_success()
            return result
        except Exception as e:
            await circuit.record_failure()
            raise

    # =========================================================================
    # Tier 2 Scaling (Aggregators)
    # =========================================================================

    async def _auto_scaler_loop(self):
        """Background loop for auto-scaling decisions"""
        while self._running:
            try:
                await self._make_scaling_decision()
            except Exception as e:
                logger.error(f"Auto-scaler error: {e}")

            await asyncio.sleep(self.config.auto_scaler_interval)

    async def _make_scaling_decision(self):
        """Make scaling decisions by delegating to self.scaler."""
        active_nodes = await self.registry.get_active_nodes()
        idle_nodes = await self.registry.get_idle_nodes()

        default_base = None
        if self.model_config:
            bases = self.model_config.list_base_models()
            if bases:
                default_base = bases[0]

        snapshot = ClusterSnapshot(
            num_active_nodes=len(active_nodes),
            num_idle_nodes=len(idle_nodes),
            min_active_nodes=self.config.min_active_nodes,
            max_active_nodes=self.config.max_active_nodes,
            pending_requests=self._pending_requests,
            default_base_model=default_base,
        )

        drainable = await self.registry.get_drainable_nodes(
            idle_threshold=self.config.node_scale_down_delay
        )
        drainable_ids = [n.node_id for n in drainable]

        decisions = self.scaler.make_scaling_decisions(snapshot, drainable_ids)

        # Execute decisions — launch scale-ups in parallel, scale-downs sequentially
        scale_ups = [d for d in decisions if d.action == ScalingAction.SCALE_UP]
        scale_downs = [d for d in decisions if d.action == ScalingAction.SCALE_DOWN]

        if scale_ups:
            coros = []
            for decision in scale_ups:
                bm = decision.base_model or default_base
                logger.info(f"Scaling up: pending={self._pending_requests}, base_model={bm}")
                coros.append(self._scale_up_node(base_model=bm))
            await asyncio.gather(*coros, return_exceptions=True)

        for decision in scale_downs:
            if decision.node_id:
                logger.info(f"Scaling down node {decision.node_id}")
                await self._scale_down_node(decision.node_id)

    async def _scale_up_node(self, base_model: Optional[str] = None) -> Optional[NodeInfo]:
        """Scale up: launch aggregator on an idle node with the specified base model"""
        node = await self.placement.select_node_for_scale_up(base_model)

        if not node:
            logger.warning(f"No idle nodes available for scale up (base_model={base_model})")
            return None

        model_id = None
        if base_model and self.model_config:
            config = self.model_config.get_base_model_config(base_model)
            if config:
                model_id = config.model_id

        logger.info(f"Scaling up node {node.node_id} with base_model={base_model}")
        await self.registry.update_node_status(node.node_id, NodeStatus.STARTING)

        if base_model:
            await self.registry.set_node_base_model(node.node_id, base_model, model_id)
            self.model_config.assign_node_to_base_model(node.node_id, base_model)

        try:
            launch_payload = {
                "force": False,
                "base_model": base_model,
                "model_id": model_id,
            }

            async with self._session.post(
                f"{node.url}/aggregator/launch",
                json=launch_payload,
                timeout=aiohttp.ClientTimeout(total=self.config.node_startup_timeout)
            ) as response:
                try:
                    result = await response.json()
                except Exception as e:
                    logger.error(f"Invalid JSON from node {node.node_id} during scale-up: {e}")
                    await self.registry.update_node_status(node.node_id, NodeStatus.FAILED)
                    return None

                if result.get("success"):
                    await self.registry.update_node_status(node.node_id, NodeStatus.ACTIVE)
                    logger.info(f"Node {node.node_id} is now active with base_model={base_model}")
                    return await self.registry.get_node(node.node_id)
                else:
                    await self.registry.update_node_status(node.node_id, NodeStatus.FAILED)
                    logger.error(f"Node {node.node_id} failed to start: {result.get('message')}")
                    return None

        except Exception as e:
            logger.error(f"Failed to scale up node {node.node_id}: {e}")
            await self.registry.update_node_status(node.node_id, NodeStatus.FAILED)
            return None

    async def _scale_down_node(self, node_id: str):
        """Scale down: stop aggregator on a node"""
        node = await self.registry.get_node(node_id)
        if not node:
            return

        logger.info(f"Scaling down node {node_id}")
        await self.registry.update_node_status(node_id, NodeStatus.DRAINING)

        try:
            async with self._session.post(
                f"{node.url}/aggregator/stop",
                json={"graceful": True, "timeout": self.config.graceful_drain_timeout},
                timeout=aiohttp.ClientTimeout(total=self.config.graceful_drain_timeout * 2)
            ) as response:
                if response.status == 200:
                    await self.registry.update_node_status(node_id, NodeStatus.IDLE)
                    logger.info(f"Node {node_id} is now idle")
                else:
                    logger.error(f"Node {node_id} stop returned {response.status}, marking FAILED")
                    await self.registry.update_node_status(node_id, NodeStatus.FAILED)

        except Exception as e:
            logger.error(f"Failed to scale down node {node_id}: {e}")
            await self.registry.update_node_status(node_id, NodeStatus.FAILED)

    # =========================================================================
    # Health Monitoring
    # =========================================================================

    async def _health_monitor_loop(self):
        """Monitor node health: detect stale nodes and recover FAILED nodes."""
        while self._running:
            try:
                # Check for stale nodes (only ACTIVE/DRAINING, not STARTING/IDLE/FAILED)
                stale_nodes = await self.registry.get_stale_nodes(
                    timeout=self.config.heartbeat_timeout
                )

                for node in stale_nodes:
                    logger.warning(f"Node {node.node_id} heartbeat timeout")
                    await self.registry.update_node_status(node.node_id, NodeStatus.FAILED)

                # Sync circuit breaker states to registry
                for node_id, circuit in self._node_circuits.items():
                    await self.registry.update_node_error_state(node_id, circuit.state.value)

                # Recover FAILED nodes that have resumed heartbeating
                await self._recover_failed_nodes()

            except Exception as e:
                logger.error(f"Health monitor error: {e}")

            await asyncio.sleep(self.config.health_check_interval)

    async def _recover_failed_nodes(self):
        """Check FAILED nodes — if they've resumed heartbeating, transition back."""
        async with self.registry._nodes_lock:
            now = time.time()
            for node in self.registry.nodes.values():
                if node.status != NodeStatus.FAILED:
                    continue
                # If heartbeat is recent (within timeout), the node agent is alive
                if now - node.last_heartbeat < self.config.heartbeat_timeout:
                    # Determine correct status based on aggregator state
                    if node.aggregator_pid:
                        new_status = NodeStatus.ACTIVE
                    else:
                        new_status = NodeStatus.IDLE
                    logger.info(f"Node {node.node_id} recovered: FAILED → {new_status.value}")
                    node.status = new_status

    # =========================================================================
    # Metrics
    # =========================================================================

    async def _metrics_loop(self):
        """Log metrics periodically"""
        while self._running:
            await asyncio.sleep(self.config.metrics_interval)
            metrics = await self._get_metrics()
            logger.info(
                f"Cluster: nodes={metrics['active_nodes']}/{metrics['total_nodes']}, "
                f"workers={metrics['total_workers']}, "
                f"requests={self._completed_requests}/{self._total_requests}, "
                f"rps={metrics['requests_per_second']:.2f}"
            )

    async def _get_metrics(self) -> dict:
        """Get cluster metrics"""
        state = await self.registry.get_cluster_state()

        # Calculate RPS
        now = time.time()
        self._requests_last_minute = [t for t in self._requests_last_minute if now - t < 60]
        rps = len(self._requests_last_minute) / 60.0

        total_workers = sum(
            n.get("active_workers", 0)
            for n in state["nodes"].values()
        )

        # Per-node breakdown
        all_node_ids = set(
            list(self._per_node_completed.keys()) +
            list(self._per_node_errors.keys()) +
            list(self._inflight_per_node.keys()) +
            list(self._per_node_latency.keys())
        )
        per_node = {
            node_id: {
                "completed": self._per_node_completed.get(node_id, 0),
                "errors": self._per_node_errors.get(node_id, 0),
                "circuit_state": (self._node_circuits[node_id].state.value
                                  if node_id in self._node_circuits else "closed"),
                "inflight": self._inflight_per_node.get(node_id, 0),
                "latency": self._percentiles_from(self._per_node_latency.get(node_id, [])),
            }
            for node_id in all_node_ids
        }

        return {
            "total_nodes": state["summary"]["total_nodes"],
            "active_nodes": state["summary"]["active_nodes"],
            "idle_nodes": state["summary"]["idle_nodes"],
            "total_workers": total_workers,
            "pending_requests": self._pending_requests,
            "total_requests": self._total_requests,
            "completed_requests": self._completed_requests,
            "failed_requests": self._failed_requests,
            "requests_per_second": rps,
            "latency": self._compute_percentiles(),
            "per_node": per_node,
        }

    async def _idempotency_cleanup_loop(self):
        """Periodically clean expired idempotency entries."""
        while self._running:
            await asyncio.sleep(self.config.idempotency_cleanup_interval)
            now = time.time()
            async with self._idempotency_lock:
                expired = [k for k, (ts, _) in self._idempotency_cache.items()
                           if now - ts > self._idempotency_ttl]
                for k in expired:
                    del self._idempotency_cache[k]
                if expired:
                    logger.debug(f"Cleaned {len(expired)} expired idempotency entries")

    # =========================================================================
    # Run
    # =========================================================================

    def run(self):
        """Run the global controller"""
        uvicorn.run(self.app, host=self.config.host, port=self.config.port)


# CLI Entry Point

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Global Controller")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=GLOBAL_CONTROLLER_PORT)
    parser.add_argument("--min-nodes", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=10)
    parser.add_argument("--scale-down-delay", type=float, default=300.0)
    parser.add_argument("--placement-policy", type=str, default="affinity",
                        help="Placement policy: affinity, least_loaded, locality, gpu_aware, binpack (default: affinity)")
    parser.add_argument("--adapters-per-model", type=int, default=50,
                        help="Adapter count for the auto-generated benchmark config, e.g. the current sweep's pool size (default: 50)")
    parser.add_argument("--adapter-prefix", type=str, default="../sim-adapters/pool-10-r16/lora-",
                        help="Adapter name prefix for the auto-generated benchmark config, e.g. ../sim-adapters/pool-<N>/lora-")

    args = parser.parse_args()

    config = GlobalControllerConfig(
        host=args.host,
        port=args.port,
        min_active_nodes=args.min_nodes,
        max_active_nodes=args.max_nodes,
        node_scale_down_delay=args.scale_down_delay,
        placement_policy=args.placement_policy,
        adapters_per_model=args.adapters_per_model,
        adapter_prefix=args.adapter_prefix,
    )

    controller = GlobalController(config)
    controller.run()
