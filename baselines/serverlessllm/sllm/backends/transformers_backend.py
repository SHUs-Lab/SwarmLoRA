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
import json
import os
import queue
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import peft
import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.generation.streamers import BaseStreamer

from sllm.backends.backend_utils import BackendStatus, SllmBackend
from sllm.logger import init_logger
from sllm_store.transformers import load_lora, load_model, save_lora

logger = init_logger(__name__)


# Guarded post-admit margin, shared with SwarmLoRA and ServerlessLoRA so the
# guards cannot drift apart.
def _load_margin():
    import importlib.util
    _p = os.path.abspath(__file__)
    for _ in range(5):          # backends -> sllm -> serverlessllm -> baselines -> root
        _p = os.path.dirname(_p)
    _p = os.path.join(_p, "src", "controller", "admission.py")
    _sp = importlib.util.spec_from_file_location("swarm_admission_margin", _p)
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    return _m
# Every system reserves the same margin from this one policy, so a tree that
# cannot load it would run a different one. Fail at import instead.
_adm = _load_margin()
_guarded_margin = _adm.post_admit_margin
_SLO_S = float(os.environ.get("ADMISSION_SLO_S", _adm.DEFAULT_SLO_S))


class DeletingException(Exception):
    pass


class InferenceStatus(BaseStreamer):
    def __init__(self, status: BackendStatus):
        super().__init__()
        self.status = status
        self.intermediate = []

    def put(self, value):
        value = value.tolist()
        # Normalize: next_tokens is 1D [batch], input_ids is 2D [batch, seq_len]
        if isinstance(value[0], int):
            # 1D tensor (next_tokens) — wrap each token in a list
            value = [[v] for v in value]
        if not self.intermediate:
            self.intermediate = value
        else:
            # NOTE: This does not support in-flight batching
            # or dynamic batch size
            for i, v in enumerate(value):
                self.intermediate[i].extend(v)
        if self.status == BackendStatus.DELETING:
            raise DeletingException("Backend is deleting")

    def end(self):
        logger.error("Inference completed")

    def get(self):
        return deepcopy(self.intermediate)

    def delete(self):
        logger.info("Deleting intermediate output")
        self.intermediate = []


class FirstTokenTimer:
    """LogitsProcessor that records the timestamp of the first forward pass
    (i.e. right after prefill completes and first token logits are produced)."""

    def __init__(self):
        self.first_token_time: Optional[float] = None

    def __call__(self, input_ids, scores):
        if self.first_token_time is None:
            self.first_token_time = time.monotonic()
        return scores


@dataclass
class BatchRequest:
    """A single request waiting to be batched."""

    request_data: Dict[str, Any]
    prompt: str
    lora_adapter_name: Optional[str]
    max_tokens: int
    temperature: float
    model_name: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    done_event: threading.Event = field(default_factory=threading.Event)
    # Timestamps for latency decomposition
    t_queue_enter: float = 0.0       # when request enters batch_queue
    t_batch_start: float = 0.0      # when batch processor picks it up
    t_generate_start: float = 0.0   # when model.generate() starts for this adapter group
    t_first_token: float = 0.0      # when first token logits are produced (after prefill)
    t_generate_end: float = 0.0     # when model.generate() ends


class TransformersBackend(SllmBackend):
    def __init__(
        self, model_name: str, backend_config: Optional[Dict[str, Any]] = None
    ) -> None:
        self.backend_config = backend_config
        logger.info(
            f"Initializing TransformersBackend for {model_name} with config: {backend_config}"
        )
        self.model_name = model_name
        self.pretrained_model_name_or_path = backend_config.get(
            "pretrained_model_name_or_path"
        )
        self.status: BackendStatus = BackendStatus.UNINITIALIZED
        self.inf_status = InferenceStatus(self.status)
        self.status_lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.past_key_values = None

        # Batching configuration
        self.batch_size = backend_config.get("batch_size", 1)
        self.batch_wait_ms = backend_config.get("batch_wait_ms", 50)
        self.batch_queue = queue.Queue() if self.batch_size > 1 else None
        self.batch_thread = None

        # EMA of the work still owed after the deadline check passes: from the
        # prune point to the first token (adapter wait + prefill). Reserving it
        # lets the check ask whether the deadline will pass before the request
        # finishes. Starts at 0 until there are observations.
        self._post_admit_s = 0.0
        self._post_admit_n = 0

    def convert_str_to_json(self, json_str):
        try:
            # Parse the JSON string and return the corresponding Python object
            json_obj = json.loads(json_str)
            return json_obj
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON string: {e}")
            return None

    def init_backend(self) -> None:
        with self.status_lock:
            if self.status != BackendStatus.UNINITIALIZED:
                return
            device_map = self.backend_config.get("device_map", "auto")
            torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
            torch_dtype = getattr(torch, torch_dtype)
            hf_model_class = self.backend_config.get("hf_model_class", None)
            if torch_dtype is None:
                logger.warning(
                    f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
                )
                torch_dtype = torch.float16
            if hf_model_class is None:
                logger.error(
                    f"hf_model_class cannot be None. Please provide a valid model class"
                )
                raise ValueError(
                    "hf_model_class cannot be None. Please provide a valid model class"
                )
            quantization_config = self.backend_config.get(
                "quantization_config", None
            )

            storage_path = os.getenv("STORAGE_PATH", "./models")
            model_path = os.path.join("transformers", self.model_name)
            self.model = load_model(
                model_path,
                device_map=device_map,
                torch_dtype=torch_dtype,
                storage_path=storage_path,
                hf_model_class=hf_model_class,
                quantization_config=quantization_config,
            )
            tokenizer_path = os.path.join(
                storage_path, "transformers", self.model_name, "tokenizer"
            )
            if os.path.exists(tokenizer_path):
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            else:
                # Fall back to load from system's cache
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.pretrained_model_name_or_path, local_files_only=True
                )

            # Configure tokenizer for batching
            if self.batch_size > 1:
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                logger.info(
                    f"Configured tokenizer for batching: "
                    f"padding_side=left, pad_token={self.tokenizer.pad_token}"
                )

            self.status = BackendStatus.RUNNING

        # Start batch processor thread (outside status_lock)
        if self.batch_size > 1 and self.batch_queue is not None:
            self.batch_thread = threading.Thread(
                target=self._batch_processor_loop, daemon=True
            )
            self.batch_thread.start()
            logger.info(
                f"Started batch processor (batch_size={self.batch_size}, "
                f"batch_wait_ms={self.batch_wait_ms})"
            )

    def _tokenize(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to("cuda:0")

    def _encoder_tokenize(self, query: str, max_length: int):
        return self.tokenizer(
            query,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:0")

    def encode(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        def last_token_pool(
            last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            left_padding = (
                attention_mask[:, -1].sum() == attention_mask.shape[0]
            )
            if left_padding:
                return last_hidden_states[:, -1]
            else:
                sequence_lengths = attention_mask.sum(dim=1) - 1
                batch_size = last_hidden_states.shape[0]
                return last_hidden_states[
                    torch.arange(batch_size, device=last_hidden_states.device),
                    sequence_lengths,
                ]

        def get_detailed_instruct(task_description: str, query: str) -> str:
            return f"Instruct: {task_description}\nQuery: {query}"

        model_name = request_data.get("model", "dummy-model")
        task_instruct = request_data.get("task_instruct", "")
        max_length = request_data.get("max_length", 4096)
        query = request_data.get("input", [])

        if not query:
            return {"error": "Missing query in request data"}

        query = [get_detailed_instruct(task_instruct, q) for q in query]

        batch_dict = self._encoder_tokenize(query, max_length)
        with torch.no_grad():
            output = self.model(**batch_dict, output_hidden_states=True)
        embeddings = last_token_pool(
            output.hidden_states[-1], batch_dict["attention_mask"]
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        query_tokens = sum([len(self.tokenizer.tokenize(q)) for q in query])
        response = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": embeddings[i].tolist(),
                }
                for i in range(len(embeddings))
            ],
            "model": model_name,
            "usage": {
                "query_tokens": query_tokens,
                "total_tokens": query_tokens,
            },
        }

        return response

    def generate(self, request_data: Optional[Dict[str, Any]]):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        assert self.model is not None

        # If batching is disabled, use the original single-request path
        if self.batch_size <= 1:
            return self._generate_single(request_data)

        # --- Batched path ---
        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)
        lora_adapter_name = request_data.get("lora_adapter_name", None)

        # Build prompt
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            prompt = "\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in messages
            )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        # Adapter validation is skipped here for batched path —
        # the batch processor will load adapters before generating.

        batch_req = BatchRequest(
            request_data=request_data,
            prompt=prompt,
            lora_adapter_name=lora_adapter_name,
            max_tokens=max_tokens,
            temperature=temperature,
            model_name=model_name,
            t_queue_enter=time.monotonic(),
        )
        self.batch_queue.put(batch_req)

        # Wait for batch processor to complete this request
        batch_req.done_event.wait()

        if batch_req.error is not None:
            raise batch_req.error

        return batch_req.result

    def _generate_single(self, request_data: Optional[Dict[str, Any]]):
        """Original single-request generation path (no batching)."""
        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)
        lora_adapter_name = request_data.get("lora_adapter_name", None)

        # Combine messages to form the prompt
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            prompt = "\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in messages
            )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "streamer": self.inf_status,
        }

        if lora_adapter_name:
            if (
                not hasattr(self.model, "peft_config")
                or lora_adapter_name not in self.model.peft_config
            ):
                return {"error": f"LoRA adapter {lora_adapter_name} not found"}
            generate_kwargs["adapter_names"] = [lora_adapter_name]

        inputs = self._tokenize(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        # Generate response
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )
        except DeletingException:
            logger.info("Backend is shutting down. Aborting request")
            output_tokens = self.inf_status.get()
            self.inf_status.delete()
            return {
                "preempted": "True",
                "current_output": output_tokens,
                "completed_tokens": len(output_tokens[0]) - prompt_tokens,
            }
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
        else:
            output_text = self.tokenizer.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True
            )
            total_tokens = len(outputs[0])
            completion_tokens = total_tokens - prompt_tokens
            finish_reason = (
                "stop" if completion_tokens < max_tokens else "length"
            )

            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            self.inf_status.delete()

            return response

    def _batch_processor_loop(self):
        """Daemon thread: drain queue, group by adapter, generate per adapter."""
        from collections import defaultdict

        logger.info("Batch processor loop started")
        while True:
            # Block until at least one request arrives
            try:
                first_req = self.batch_queue.get(timeout=1.0)
            except queue.Empty:
                with self.status_lock:
                    if self.status in (
                        BackendStatus.STOPPING,
                        BackendStatus.DELETING,
                    ):
                        logger.info("Batch processor loop stopping")
                        return
                continue

            # Put first_req back so the drain loop handles it uniformly
            pending: List[BatchRequest] = [first_req]

            # Process one adapter group at a time, then re-drain.
            # This ensures stale requests are pruned between groups
            # and new arrivals get a chance to be picked up.
            while pending or not self.batch_queue.empty():
                # Drain new arrivals into pending
                deadline = time.monotonic() + self.batch_wait_ms / 1000.0
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        pending.append(
                            self.batch_queue.get(timeout=remaining)
                        )
                    except queue.Empty:
                        break
                while not self.batch_queue.empty():
                    try:
                        pending.append(self.batch_queue.get_nowait())
                    except queue.Empty:
                        break

                if not pending:
                    break

                # Prune stale requests
                now = time.monotonic()
                now_wall = time.time()
                live = []
                for req in pending:
                    req.t_batch_start = now
                    wait = now - req.t_queue_enter
                    # The router stamps a shared absolute deadline covering
                    # BOTH queueing stages, so the two together cannot exceed
                    # one SLO budget. Wall clock, because the deadline was set
                    # in the router's process and monotonic clocks are not
                    # comparable across processes. A caller that bypasses the
                    # router leaves it unset; bound this stage by the same SLO
                    # then, so the wait is never unbounded.
                    slo_deadline = req.request_data.get("_slo_deadline_wall")
                    # Reserve the work still owed after this check.
                    _margin = _guarded_margin(self._post_admit_s,
                                              self._post_admit_n, _SLO_S)
                    if slo_deadline is not None:
                        expired = (now_wall + _margin) > slo_deadline
                        limit = "slo deadline"
                    else:
                        expired = (wait + _margin) > _SLO_S
                        limit = f"max {_SLO_S}s"
                    if expired:
                        logger.warning(
                            f"Dropping request after {wait:.1f}s ({limit})"
                        )
                        req.error = TimeoutError(
                            f"Request waited {wait:.1f}s ({limit})"
                        )
                        req.done_event.set()
                    else:
                        live.append(req)
                pending = live
                if not pending:
                    break

                # Group by adapter, pick whichever group's OLDEST pending
                # request has waited longest -- not whichever group is
                # currently biggest. Backlog size isn't a fair priority
                # signal here: for the same arrival rate, an adapter whose
                # completions are naturally slower accumulates a bigger
                # backlog than a fast one (Little's law), so "biggest group
                # wins" keeps re-selecting the slow adapter every cycle while
                # the fast adapter's requests just sit aging toward the queue
                # timeout without ever winning the comparison. Oldest-first
                # removes that starvation.
                adapter_groups: Dict[str, List[BatchRequest]] = (
                    defaultdict(list)
                )
                for req in pending:
                    key = req.lora_adapter_name or "__base__"
                    adapter_groups[key].append(req)

                best_key = min(
                    adapter_groups,
                    key=lambda k: min(r.t_queue_enter for r in adapter_groups[k]),
                )
                group = adapter_groups.pop(best_key)

                # Remaining requests go back to pending for next cycle
                pending = []
                for reqs in adapter_groups.values():
                    pending.extend(reqs)

                logger.info(
                    f"Processing {len(group)} requests for "
                    f"{best_key}, {len(pending)} pending"
                )

                # Load adapter if needed
                if best_key != "__base__":
                    if (
                        not hasattr(self.model, "peft_config")
                        or best_key not in self.model.peft_config
                    ):
                        adapter_path = self.backend_config.get(
                            "lora_adapters", {}
                        ).get(best_key)
                        if adapter_path:
                            logger.info(
                                f"Loading adapter {best_key} "
                                f"(path={adapter_path})"
                            )
                            try:
                                self.load_lora_adapter(
                                    lora_name=best_key,
                                    lora_path=adapter_path,
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to load adapter "
                                    f"{best_key}: {e}"
                                )
                                for req in group:
                                    if not req.done_event.is_set():
                                        req.error = e
                                        req.done_event.set()
                                continue

                # Process in sub-batches
                for i in range(0, len(group), self.batch_size):
                    sub_batch = group[i : i + self.batch_size]
                    try:
                        self._execute_adapter_group(
                            best_key, sub_batch
                        )
                    except Exception as e:
                        logger.error(
                            f"Generate failed for adapter "
                            f"{best_key}: {e}"
                        )
                        for req in sub_batch:
                            if not req.done_event.is_set():
                                req.error = e
                                req.done_event.set()

    def _execute_adapter_group(self, adapter_key: str, group: List[BatchRequest]):
        """Run generate() for a group of requests sharing the same adapter."""
        # Switch adapter (fast path via set_adapter)
        if adapter_key != "__base__":
            self.model.set_adapter(adapter_key)
        else:
            if hasattr(self.model, "disable_adapter"):
                # Will be used as context manager below
                pass

        prompts = [req.prompt for req in group]
        max_tokens = max(req.max_tokens for req in group)
        temperature = group[0].temperature

        # Batch tokenize with left-padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to("cuda:0")

        padded_input_len = inputs.input_ids.shape[1]

        # Track actual (non-padded) prompt lengths per request
        prompt_token_counts = []
        for i in range(len(group)):
            actual_tokens = inputs["attention_mask"][i].sum().item()
            prompt_token_counts.append(int(actual_tokens))

        # Hook to capture first token time (after prefill)
        first_token_timer = FirstTokenTimer()

        generate_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "logits_processor": [first_token_timer],
        }

        # Stamp generate start
        t_gen_start = time.monotonic()
        for req in group:
            req.t_generate_start = t_gen_start

        try:
            with torch.no_grad():
                if adapter_key == "__base__" and hasattr(self.model, "disable_adapter"):
                    with self.model.disable_adapter():
                        outputs = self.model.generate(
                            **inputs, **generate_kwargs
                        )
                else:
                    outputs = self.model.generate(
                        **inputs, **generate_kwargs
                    )
        except Exception as e:
            logger.error(f"model.generate() failed for adapter {adapter_key}: {e}")
            for req in group:
                req.error = e
                req.done_event.set()
            return

        # Stamp generate end and first token time
        t_gen_end = time.monotonic()
        t_first = first_token_timer.first_token_time or t_gen_start
        for req in group:
            req.t_generate_end = t_gen_end
            req.t_first_token = t_first

        # Distribute results to individual requests
        for i, req in enumerate(group):
            try:
                generated_tokens = outputs[i][padded_input_len:]
                output_text = self.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )

                prompt_tokens = prompt_token_counts[i]
                completion_tokens = len(generated_tokens)
                total_tokens = prompt_tokens + completion_tokens
                finish_reason = (
                    "stop" if completion_tokens < req.max_tokens else "length"
                )

                # Latency decomposition (seconds)
                queue_time = req.t_batch_start - req.t_queue_enter
                adapter_wait = req.t_generate_start - req.t_batch_start
                prefill_time = req.t_first_token - req.t_generate_start
                # Post-admit margin: prune point (t_batch_start) -> first token.
                if req.t_first_token and req.t_batch_start:
                    _obs = max(0.0, req.t_first_token - req.t_batch_start)
                    self._post_admit_s = (
                        _obs if self._post_admit_n == 0
                        else 0.2 * _obs + 0.8 * self._post_admit_s)
                    self._post_admit_n += 1
                decode_time = req.t_generate_end - req.t_first_token
                generate_time = req.t_generate_end - req.t_generate_start

                response = {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": req.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": output_text,
                            },
                            "logprobs": None,
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    "timings": {
                        "queue_time_s": round(queue_time, 4),
                        "adapter_wait_s": round(adapter_wait, 4),
                        "prefill_time_s": round(prefill_time, 4),
                        "decode_time_s": round(decode_time, 4),
                        "generate_time_s": round(generate_time, 4),
                    },
                }
                req.result = response
            except Exception as e:
                logger.error(f"Failed to process batch item {i}: {e}")
                req.error = e
            finally:
                req.done_event.set()

    def load_lora_adapter(self, lora_name: str, lora_path: str):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        if (
            hasattr(self.model, "peft_config")
            and lora_name in self.model.peft_config
        ):
            logger.info(f"LoRA adapter {lora_name} already loaded")
            return

        # Use lora_name for storage path (not lora_path which may be absolute)
        lora_path = os.path.join("transformers", lora_name)
        storage_path = os.getenv("STORAGE_PATH", "./models")
        device_map = self.backend_config.get("device_map", "auto")
        torch_dtype = self.backend_config.get("torch_dtype", torch.float16)
        torch_dtype = getattr(torch, torch_dtype)
        if torch_dtype is None:
            logger.warning(
                f"Invalid torch_dtype: {torch_dtype}. Using torch.float16"
            )
            torch_dtype = torch.float16
        self.model = load_lora(
            self.model,
            lora_name,
            lora_path,
            device_map=device_map,
            storage_path=storage_path,
            torch_dtype=torch_dtype,
        )
        logger.info(f"Loaded LoRA adapter {lora_name} from {lora_path}")

    def shutdown(self):
        """Abort all requests and shutdown the backend."""
        with self.status_lock:
            if self.status == BackendStatus.DELETING:
                return
            self.status = BackendStatus.DELETING
            if self.inf_status:
                self.inf_status.status = BackendStatus.DELETING

        # Drain the batch queue and signal errors to waiting requests
        if self.batch_queue is not None:
            while not self.batch_queue.empty():
                try:
                    req = self.batch_queue.get_nowait()
                    req.error = Exception("Backend is shutting down")
                    req.done_event.set()
                except queue.Empty:
                    break

        # Wait for batch thread to finish
        if self.batch_thread is not None and self.batch_thread.is_alive():
            self.batch_thread.join(timeout=5.0)

        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)

        if self.model is not None:
            del self.model

    def stop(self) -> None:
        """Wait for all requests to finish and shutdown the backend."""
        with self.status_lock:
            if self.status.value >= BackendStatus.STOPPING.value:
                return
            self.status = BackendStatus.STOPPING
        while self.inf_status and len(self.inf_status.get()) > 0:
            logger.info("Waiting for all requests to finish")
            time.sleep(1)
        logger.info("All requests finished. Shutting down the backend.")
        self.shutdown()

    def get_current_tokens(self) -> List[List[int]]:
        """Return a list of all ongoing request tokens."""
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return []

        status = self.inf_status.get()
        logger.info(f"Current tokens: {status}")
        return status

    def resume_kv_cache(self, request_datas):
        logger.info(f"Resuming cache for {request_datas}")
        with torch.no_grad():
            device = self.model.device
            input_ids = torch.tensor(request_datas).to(device)
            logger.info(input_ids)
            output = self.model.generate(
                input_ids,
                past_key_values=self.past_key_values,
                max_new_tokens=1,
                return_dict_in_generate=True,
                return_legacy_cache=True,
            )
            self.past_key_values = output.past_key_values
            self.current_tokens = output.sequences
        logger.info(f"Resumed {len(self.past_key_values[0][0][0][0])} tokens")

    def resume_generate(
        self, request_data: Optional[Dict[str, Any]], current_output
    ):
        with self.status_lock:
            if self.status != BackendStatus.RUNNING:
                return {"error": "Model not initialized"}

        assert self.model is not None

        model_name = request_data.get("model", "dummy-model")
        messages = request_data.get("messages", [])
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 10)

        # Combine messages to form the prompt
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            prompt = "\n".join(
                f"{message['role'].capitalize()}: {message['content']}"
                for message in messages
            )

        if not prompt:
            return {"error": "Missing prompt in request data"}

        inputs = self._tokenize(prompt)
        prompt_tokens = inputs.input_ids.shape[1]

        # Generate response
        try:
            with torch.no_grad():
                device = self.model.device
                current_output = torch.tensor(current_output).to(device)
                if len(current_output[0]) < len(self.current_tokens[0]):
                    current_output = self.current_tokens
                outputs = self.model.generate(
                    current_output,
                    past_key_values=self.past_key_values,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    streamer=self.inf_status,
                )
        except DeletingException:
            logger.error("Backend is shutting down. Aborting request")
            raise DeletingException("Backend is shutting down")
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise e
        else:
            output_text = self.tokenizer.decode(
                outputs[0][prompt_tokens:], skip_special_tokens=True
            )
            total_tokens = len(outputs[0])
            completion_tokens = total_tokens - prompt_tokens
            # FIXME: consider corner case when max_tokens is reached
            finish_reason = (
                "stop" if completion_tokens < max_tokens else "length"
            )

            # Generate response compatible with OpenAI's API
            response = {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": output_text,
                        },
                        "logprobs": None,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }

            self.inf_status.delete()

            return response
