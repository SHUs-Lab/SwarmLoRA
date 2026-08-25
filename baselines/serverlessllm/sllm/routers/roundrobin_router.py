# ---------------------------------------------------------------------------- #
#  serverlessllm                                                               #
#  copyright (c) serverlessllm team 2024                                       #
#                                                                              #
#  licensed under the apache license, version 2.0 (the "license");             #
#  you may not use this file except in compliance with the license.            #
#                                                                              #
#  you may obtain a copy of the license at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/license-2.0                  #
#                                                                              #
#  unless required by applicable law or agreed to in writing, software         #
#  distributed under the license is distributed on an "as is" basis,           #
#  without warranties or conditions of any kind, either express or implied.    #
#  see the license for the specific language governing permissions and         #
#  limitations under the license.                                              #
# ---------------------------------------------------------------------------- #
import asyncio
import logging
import os
import time
import uuid
from typing import Dict, List, Optional

import ray

from sllm.fine_tuning_instance import start_ft_instance
from sllm.inference_instance import start_instance
from sllm.logger import init_logger

from ..utils import InstanceHandle
from .router_utils import SllmRouter

logger = init_logger(__name__)

# Key carrying the shared absolute (wall-clock) deadline into the backend.
# ServerlessLLM queues a request twice -- here for an instance, then again in
# that instance's batch queue -- and the two stages run in different processes.
# Giving each stage its own SLO-derived budget would let a request queue for
# nearly twice the target, so the router computes ONE deadline from arrival and
# the backend prunes against that same instant. Wall clock, not monotonic,
# because the two stages are separate Ray actors.
SLO_DEADLINE_KEY = "_slo_deadline_wall"


# Loaded from the repo's single canonical copy rather than vendored, so all
# three systems provably run the same policy. Loaded by path because this tree
# is a separate package with its own import root.
def _load_admission():
    import importlib.util

    path = os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                )
            )
        ),
        "src",
        "controller",
        "admission.py",
    )
    spec = importlib.util.spec_from_file_location("swarm_admission", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# All three systems share this policy, so a tree that cannot load it would run
# a different one and mislabel the results. Fail at import instead.
_admission_mod = _load_admission()


async def auto_scaler(
    auto_scaling_metrics: Dict[str, int], auto_scaling_config: Dict[str, int]
) -> int:
    """
    Returns desired number of instances for a model based on the auto-scaling policy
    """

    request_count = auto_scaling_metrics.get("request_count", 0)

    min_instances = auto_scaling_config.get("min_instances", 0)
    max_instances = auto_scaling_config.get("max_instances", 10)
    target_ongoing_requests = auto_scaling_config.get("target", 2)

    desired_instances = (
        request_count + target_ongoing_requests - 1
    ) // target_ongoing_requests
    desired_instances = min(
        max_instances, max(min_instances, desired_instances)
    )

    return desired_instances


class RoundRobinRouter(SllmRouter):
    def __init__(
        self,
        model_name: str,
        resource_requirements: Dict[str, int],
        backend: str,
        backend_config: Dict,
        router_config: Dict,
        enable_lora: bool = False,
        lora_adapters: Optional[Dict[str, str]] = None,
    ) -> None:
        self.model_name = model_name
        self.resource_requirements = resource_requirements
        self.backend = backend
        self.backend_config = backend_config
        self.router_config = router_config

        self.loop_interval = 1
        self.loop = asyncio.get_running_loop()
        self.request_queue = asyncio.Queue()  # type:ignore
        # Inference instance pools
        self.starting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_inference_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        # Fine-tuning instance pools
        self.starting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.deleting_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.ready_ft_instances: Dict[str, InstanceHandle] = {}  # type:ignore
        self.instance_management_lock = asyncio.Lock()

        # Request deadline: each request has until arrival+SLO, enforced here
        # and again in the backend's batch queue via SLO_DEADLINE_KEY.
        self._admission = _admission_mod.AdmissionConfig.from_env()

        # E1: measured work remaining after an instance is assigned (backend
        # batch queue + adapter wait + prefill). Drives the dynamic stage-1
        # deadline. Note this router assigns eagerly, so in practice a request
        # rarely waits here at all -- measured stage-1 drops were 0 of 1761.
        self._e1_s = 0.0
        self._e1_n = 0

        self.auto_scaling_config = {}
        self.auto_scaling_lock = asyncio.Lock()

        self.request_count = 0
        self.request_count_lock = asyncio.Lock()

        self.fine_tuning_count = 0
        self.fine_tuning_count_lock = asyncio.Lock()

        self.running = False
        self.running_lock = asyncio.Lock()

        self.idle_time = 0
        self.idle_time_lock = asyncio.Lock()

        self.enable_lora = enable_lora
        self.loaded_lora_adapters = lora_adapters
        self.lora_lock = asyncio.Lock()

        self.auto_scaler = None
        logger.info(f"Created new handler for model {self.model_name}")

    async def start(
        self, auto_scaling_config: Dict[str, int], mode: str = "inference"
    ):
        self.model_loading_scheduler = ray.get_actor("model_loading_scheduler")
        if mode == "inference":
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config
            self.auto_scaler = asyncio.create_task(self._auto_scaler_loop())
            self.load_balancer = asyncio.create_task(self._load_balancer_loop())
        async with self.running_lock:
            self.running = True
        logger.info(f"Started handler for model {self.model_name}")

    async def update(
        self,
        auto_scaling_config: Optional[Dict[str, int]] = None,
        lora_adapters: Optional[Dict[str, str]] = None,
    ):
        if auto_scaling_config is not None:
            async with self.auto_scaling_lock:
                self.auto_scaling_config = auto_scaling_config

        if lora_adapters is not None:
            async with self.lora_lock:
                self.loaded_lora_adapters = lora_adapters

        logger.info(
            f"Model {self.model_name}'s auto scaling config updated to {auto_scaling_config}"
        )

    def _new_instance_id(self):
        pattern = "{model_name}_{id}"
        return pattern.format(model_name=self.model_name, id=uuid.uuid4())

    async def inference(self, request_data: dict, action: str):
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.request_count_lock:
            self.request_count += 1

        async with self.idle_time_lock:
            self.idle_time = 0

        # Anchor the deadline at arrival so both queueing stages share it.
        arrival_wall = time.time()
        budget = self._admission.slo_s

        instance_allocation = self.loop.create_future()
        # Absolute deadline carried on the future so the allocation loop can
        # re-check it as conditions change, matching how SwarmLoRA and
        # ServerlessLoRA re-evaluate on every pass. A one-shot timeout computed
        # here would freeze E1 at arrival and miss congestion that develops
        # while the request waits.
        instance_allocation._slo_deadline = arrival_wall + budget
        # The allocation loop sets _cold on this future if it has to wait for
        # an instance to come up, so the capacity signal can exclude it.
        instance_allocation._cold = False
        await self.request_queue.put(instance_allocation)
        logger.info(f"Enqueued {action} request for model {self.model_name}")

        try:
            instance_id = await asyncio.wait_for(
                instance_allocation, timeout=budget
            )
        except asyncio.TimeoutError:
            instance_allocation.cancel()
            async with self.request_count_lock:
                self.request_count -= 1
            logger.warning(
                f"Queue timeout ({int(budget * 1000)}ms) for {action} "
                f"request on model {self.model_name}"
            )
            return {"error": f"Queue timeout ({int(budget * 1000)}ms)"}

        # When the instance was actually obtained; the gap from arrival is the
        # router-side queue wait, which the backend timings do not include.
        alloc_wall = time.time()

        if instance_id is None:
            async with self.request_count_lock:
                self.request_count -= 1
            logger.warning(
                f"Queue timeout (deadline) for {action} on {self.model_name}"
            )
            return {"error": f"Queue timeout ({int(budget * 1000)}ms)"}

        async with self.instance_management_lock:
            if instance_id not in self.ready_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return {"error": "Instance not found"}
            instance = self.ready_inference_instances[instance_id]

        # sanity check
        if self.enable_lora and "lora_adapter_name" in request_data:
            lora_adapter_name = request_data["lora_adapter_name"]
            if lora_adapter_name not in self.loaded_lora_adapters:
                logger.error(f"Lora adapter {lora_adapter_name} not found")
                return {"error": f"Lora adapter {lora_adapter_name} not found"}
            # For batched backends, skip pre-loading here — the batch
            # processor will load all needed adapters in one go before
            # calling model.generate(), which avoids staggering request
            # arrival into the batch queue.
            if self.backend_config.get("batch_size", 1) <= 1:
                await instance.backend_instance.load_lora_adapter.remote(
                    lora_name=lora_adapter_name,
                    lora_path=self.loaded_lora_adapters[lora_adapter_name],
                )
        # NOTE: `.remote(request_data)` does not work, don't know why.
        # Looks like a known issue:
        # https://github.com/ray-project/ray/issues/26283#issuecomment-1780691475
        # Hand the remainder of the same budget to the backend's batch queue,
        # so the two stages together cannot exceed it.
        request_data = dict(request_data)
        request_data[SLO_DEADLINE_KEY] = arrival_wall + budget

        if action == "generate":
            result = await instance.backend_instance.generate.remote(
                request_data=request_data
            )
        elif action == "encode":
            result = await instance.backend_instance.encode.remote(
                request_data=request_data
            )
        else:
            result = {"error": "Invalid action"}


        # Feed E1: everything owed after assignment, from the backend timings.
        if isinstance(result, dict) and result.get("timings"):
            _t = result["timings"]
            _obs = (_t.get("queue_time_s", 0) + _t.get("adapter_wait_s", 0)
                    + _t.get("prefill_time_s", 0))
            if 0 < _obs < self._admission.slo_s:
                self._e1_s = (_obs if self._e1_n == 0
                              else 0.2 * _obs + 0.8 * self._e1_s)
                self._e1_n += 1

        logger.info(f"Finished processing request")
        await instance.add_requests(-1)
        async with self.request_count_lock:
            self.request_count -= 1
        return result

    async def fine_tuning(self, request_data: dict):
        logger.info(f"Starting fine-tuning for model {self.model_name}")
        async with self.running_lock:
            if not self.running:
                return {"error": "Instance stopped"}

        async with self.fine_tuning_count_lock:
            self.fine_tuning_count += 1

        try:
            instance_id = await self._create_ft_instance()
        except Exception as e:
            logger.error(f"Failed to create fine-tuning instance: {str(e)}")
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {"error": f"Failed to create fine-tuning instance: {str(e)}"}

        max_wait_time = 300
        wait_time = 0
        while wait_time < max_wait_time:
            async with self.instance_management_lock:
                if instance_id in self.ready_ft_instances:
                    instance = self.ready_ft_instances[instance_id]
                    break
                elif instance_id not in self.starting_ft_instances:
                    logger.error(
                        f"Fine tuning instance {instance_id} not found in starting or ready instances"
                    )
                    async with self.fine_tuning_count_lock:
                        self.fine_tuning_count -= 1
                    return {"error": "Fine tuning instance not found"}
            await asyncio.sleep(0.1)
            wait_time += 0.1
        else:
            logger.error(
                f"Timeout waiting for fine tuning instance {instance_id} to be ready"
            )
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            return {
                "error": "Timeout waiting for fine tuning instance to be ready"
            }

        try:
            logger.info(f"Calling fine_tuning method on instance {instance_id}")
            result = await instance.backend_instance.fine_tuning.remote(
                request_data=request_data
            )

            logger.info(f"Finished processing fine-tuning {self.model_name}")
            await instance.add_requests(-1)

            await self._shutdown_instance(instance_id, is_ft=True)

            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1

            return result
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            await instance.add_requests(-1)
            await self._shutdown_instance(instance_id, is_ft=True)
            async with self.fine_tuning_count_lock:
                self.fine_tuning_count -= 1
            logger.info(
                f"Fine-tuning failed and cleaned up for model {self.model_name}"
            )
            return {"error": f"Fine-tuning failed: {str(e)}"}

    async def delete_adapters(self, lora_adapters: List[str]):
        async with self.lora_lock:
            for adapter_name in lora_adapters:
                if adapter_name in self.loaded_lora_adapters:
                    del self.loaded_lora_adapters[adapter_name]
        logger.info(
            f"Deleted LoRA adapters {lora_adapters} on model {self.model_name}"
        )

    async def shutdown(self):
        async with self.running_lock:
            self.running = False
        # stop all inference instances
        # return all unfinished requests
        while not self.request_queue.empty():
            request_data, done_event = await self.request_queue.get()
            done_event.set_result({"error": "Instance cancelled"})

        async with self.instance_management_lock:
            deleted_instance_id = list(self.ready_inference_instances.keys())
        delete_tasks = [
            self._shutdown_instance(instance_id)
            for instance_id in deleted_instance_id
        ]
        await asyncio.gather(*delete_tasks)

        return deleted_instance_id

    async def _load_balancer_loop(self):
        # Batch-draining round-robin load balancer
        round_robin_index = 0
        while True:
            # Block until at least one request arrives
            first_alloc = await self.request_queue.get()
            pending = [first_alloc]

            # Drain all currently queued requests (non-blocking)
            while not self.request_queue.empty():
                try:
                    pending.append(self.request_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(pending) > 1:
                logger.info(
                    f"Load balancer batch-assigning {len(pending)} requests "
                    f"for model {self.model_name}"
                )

            # Assign all pending requests to instances
            for alloc in pending:
                # Client already gave up (queue timeout fired in inference())
                # while this allocation was sitting in the queue — don't spend
                # an instance slot on a request nobody's waiting for anymore.
                if alloc.done():
                    continue
                allocated = False
                while not allocated:
                    if alloc.done():
                        break
                    async with self.instance_management_lock:
                        instance_options = list(
                            self.ready_inference_instances.keys()
                        )
                    # Wait for at least one ready instance
                    while not instance_options:
                        # Dynamic deadline, re-evaluated: drop once the work
                        # still owed after assignment no longer fits.
                        _dl = getattr(alloc, "_slo_deadline", None)
                        if _dl is not None and not alloc.done():
                            _e1 = _admission_mod.post_admit_margin(
                                self._e1_s, self._e1_n, self._admission.slo_s)
                            if (time.time() + _e1) > _dl:
                                alloc.set_result(None)
                                break
                        # No instance ready: this request is waiting out a cold
                        # start, so mark it for exclusion from the capacity
                        # signal (startup is not congestion).
                        alloc._cold = True
                        await asyncio.sleep(0.5)
                        if alloc.done():
                            break
                        async with self.instance_management_lock:
                            instance_options = list(
                                self.ready_inference_instances.keys()
                            )
                    if alloc.done() or not instance_options:
                        break

                    instance_id = instance_options[
                        round_robin_index % len(instance_options)
                    ]
                    round_robin_index += 1
                    async with self.instance_management_lock:
                        if instance_id not in self.ready_inference_instances:
                            continue
                        instance = self.ready_inference_instances[instance_id]
                        if await instance.check_request_queue():
                            allocated = await instance.add_requests(1)
                            if allocated:
                                if alloc.done():
                                    # Timed out between the check above and
                                    # here — release the reservation instead
                                    # of resolving an already-cancelled future.
                                    await instance.add_requests(-1)
                                else:
                                    alloc.set_result(instance_id)
                        else:
                            logger.info(
                                f"Instance {instance_id} cannot add another request"
                            )
                    if not allocated:
                        # Brief yield — don't block other assignments
                        await asyncio.sleep(0.05)

    async def _auto_scaler_loop(self):
        while True:
            # logger.info(f"Auto-scaling for model {self.model_name}")
            async with self.auto_scaling_lock:
                auto_scaling_config = self.auto_scaling_config.copy()
            async with self.request_count_lock:
                request_count = self.request_count
            auto_scaling_metrics = {"request_count": request_count}
            desired_instances = await auto_scaler(
                auto_scaling_metrics, auto_scaling_config
            )
            async with self.instance_management_lock:
                num_running_instances = len(
                    self.starting_inference_instances
                ) + len(self.ready_inference_instances)
            logger.info(
                f"{self.model_name}: {num_running_instances} instances,"
                f"need {desired_instances} instances",
            )
            if desired_instances > num_running_instances:
                instances_to_create = desired_instances - num_running_instances
                logger.info(f"Creating {instances_to_create} new instance(s)")
                for _ in range(instances_to_create):
                    await self._create_instance()
            elif desired_instances < num_running_instances:
                keep_alive = auto_scaling_config.get("keep_alive", 0)
                if self.idle_time >= keep_alive:
                    logger.info(
                        f"Stopping instance, idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    await self._stop_instance()
                    async with self.idle_time_lock:
                        self.idle_time = 0
                else:
                    logger.info(
                        f"idle_time: {self.idle_time}, keep_alive: {keep_alive}"
                    )
                    async with self.idle_time_lock:
                        self.idle_time += self.loop_interval
            else:
                # logger.info("No scaling needed")
                pass
            await asyncio.sleep(self.loop_interval)

    async def _create_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}"
        )
        # For batched backends, use a large queue so requests flow through
        # to the batch processor without being throttled at the router.
        batch_size = self.backend_config.get("batch_size", 1)
        if batch_size > 1:
            max_request_length = self.auto_scaling_config.get(
                "max_queue_length", 1000
            )
        else:
            max_request_length = self.auto_scaling_config.get(
                "max_queue_length",
                max(batch_size, self.auto_scaling_config.get("target", 1)),
            )
        logger.info(
            f"Creating new instance {instance_id} for model {self.model_name}, max queue length is {max_request_length}"
        )
        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=max_request_length,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_inference_instances[instance_id] = instance
        self.loop.create_task(self._start_instance(instance_id))

        return instance_id

    async def _create_ft_instance(self):
        instance_id = self._new_instance_id()
        logger.info(
            f"Creating new FT instance {instance_id} for model {self.model_name}"
        )

        instance = InstanceHandle(
            instance_id=instance_id,
            max_queue_length=1,
            num_gpu=self.resource_requirements["num_gpus"],
        )
        async with self.instance_management_lock:
            self.starting_ft_instances[instance_id] = instance
        self.loop.create_task(self._start_ft_instance(instance_id))
        logger.info(f"Created task for starting FT instance {instance_id}")
        return instance_id

    async def _start_instance(self, instance_id):
        async with self.instance_management_lock:
            if instance_id not in self.starting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.starting_inference_instances[instance_id]
        # Now ask model loading scheduler to load the model
        logger.info(
            f"Allocating resources for model {self.model_name} on instance {instance_id}"
        )
        startup_node = (
            await self.model_loading_scheduler.allocate_resource.remote(
                self.model_name, instance_id, self.resource_requirements
            )
        )
        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }
        logger.info(f"Startup config: {startup_config}, {self.backend_config}")

        await start_instance.options(
            resources={
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            }
        ).remote(
            instance_id,
            self.backend,
            self.model_name,
            self.backend_config,
            startup_config,
        )
        logger.info(
            f"Started instance {instance_id} for model {self.model_name}"
        )
        instance.backend_instance = ray.get_actor(instance_id)
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        await instance.backend_instance.init_backend.remote()

        async with self.instance_management_lock:
            self.ready_inference_instances[instance_id] = instance
            self.starting_inference_instances.pop(instance_id)
        return instance_id

    async def _start_ft_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.starting_ft_instances:
                logger.error(f"FT Instance {instance_id} not found")
                return
            instance = self.starting_ft_instances[instance_id]

        logger.info(
            f"Allocating FT resources for model {self.model_name} on {instance_id}"
        )
        try:
            startup_node = (
                await self.model_loading_scheduler.allocate_resource.remote(
                    self.model_name, instance_id, self.resource_requirements
                )
            )
            logger.debug(
                f"Allocated resources on node {startup_node} for FT instance {instance_id}"
            )
        except Exception as e:
            logger.error(
                f"Failed to allocate resources for FT instance {instance_id}: {str(e)}"
            )
            raise

        startup_config = {
            "num_cpus": self.resource_requirements["num_cpus"],
            "num_gpus": self.resource_requirements["num_gpus"],
            "resources": {
                "worker_node": 0.1,
                f"worker_id_{startup_node}": 0.1,
            },
        }

        try:
            instance.backend_instance = await start_ft_instance.options(
                resources=startup_config["resources"]
            ).remote(
                instance_id,
                self.backend,
                self.model_name,
                self.backend_config,
                startup_config,
            )
        except Exception as e:
            logger.error(
                f"Failed to create Ray actor for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise
        async with instance.lock:
            instance.ready = True
            instance.node_id = startup_node
        try:
            await instance.backend_instance.init_backend.remote()
        except Exception as e:
            logger.error(
                f"Failed to initialize backend for fine-tuning instance {instance_id}: {str(e)}"
            )
            raise

        async with self.instance_management_lock:
            self.ready_ft_instances[instance_id] = instance
            self.starting_ft_instances.pop(instance_id)
        logger.info(f"Fine-tuning instance {instance_id} is now ready")
        return instance_id

    async def _stop_instance(self, instance_id: Optional[str] = None):
        while len(self.ready_inference_instances) <= 0:
            await asyncio.sleep(1)

        async with self.instance_management_lock:
            if instance_id is None:
                instance_id, instance = self.ready_inference_instances.popitem()
            elif instance_id in self.ready_inference_instances:
                instance = self.ready_inference_instances.pop(instance_id)
            else:
                logger.error(f"Instance {instance_id} not found")
                return
            self.deleting_inference_instances[instance_id] = instance
        logger.info(
            f"Stopping instance {instance_id} for model {self.model_name}"
        )
        self.loop.create_task(self._finish_instance(instance_id))

    async def _finish_instance(self, instance_id: str):
        async with self.instance_management_lock:
            if instance_id not in self.deleting_inference_instances:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = self.deleting_inference_instances.pop(instance_id)
        async with instance.lock:
            instance.status = False
        await instance.backend_instance.stop.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )

    async def _shutdown_instance(self, instance_id: str, is_ft: bool = False):
        logger.info(
            f"Force deleting an instance (even if it is busy) for model {self.model_name}"
        )
        async with self.instance_management_lock:
            if is_ft:
                pool = self.ready_ft_instances
            else:
                pool = self.ready_inference_instances
            if instance_id not in pool:
                logger.error(f"Instance {instance_id} not found")
                return
            instance = pool.pop(instance_id)
            async with instance.lock:
                instance.status = False
        await instance.backend_instance.shutdown.remote()
        ray.kill(instance.backend_instance)
        await self.model_loading_scheduler.deallocate_resource.remote(
            self.model_name, instance_id, self.resource_requirements
        )
        return
