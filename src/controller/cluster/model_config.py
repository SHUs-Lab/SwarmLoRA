#  Model Configuration - Static mapping of base models and adapters           #
#  For paper-level benchmarking with multiple base models                     #

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import yaml
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class BaseModelConfig:
    """Configuration for a base model."""
    name: str                           # Short name: "llama-7b"
    model_id: str                       # HuggingFace ID: "meta-llama/Llama-2-7b-hf"
    adapters: List[str]                 # List of adapter names for this base
    adapter_paths: Dict[str, str] = field(default_factory=dict)  # adapter_name → path

    # Optional settings
    gpu_memory_gb: float = 16.0         # Expected GPU memory usage
    max_workers: int = 32               # Max workers per aggregator


@dataclass
class ClusterModelConfig:
    """Static configuration for multi-model cluster."""

    base_models: Dict[str, BaseModelConfig] = field(default_factory=dict)

    # Reverse lookup: adapter → base_model_name
    _adapter_to_base: Dict[str, str] = field(default_factory=dict)

    # Node assignments: base_model_name → list of node_ids
    node_assignments: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self):
        self._build_reverse_lookup()

    def _build_reverse_lookup(self):
        """Build adapter → base_model lookup."""
        self._adapter_to_base = {}
        for base_name, config in self.base_models.items():
            for adapter in config.adapters:
                if adapter in self._adapter_to_base:
                    logger.warning(
                        f"Adapter '{adapter}' mapped to multiple base models: "
                        f"{self._adapter_to_base[adapter]} and {base_name}"
                    )
                self._adapter_to_base[adapter] = base_name

    def get_base_model_for_adapter(self, adapter_name: str) -> Optional[str]:
        """Get the base model name for an adapter."""
        return self._adapter_to_base.get(adapter_name)

    def get_base_model_config(self, base_name: str) -> Optional[BaseModelConfig]:
        """Get configuration for a base model."""
        return self.base_models.get(base_name)

    def get_nodes_for_base_model(self, base_name: str) -> List[str]:
        """Get node IDs assigned to a base model."""
        return self.node_assignments.get(base_name, [])

    def assign_node_to_base_model(self, node_id: str, base_name: str):
        """Assign a node to run a specific base model."""
        # Validate base_name exists (warn if not, but still allow for dynamic configs)
        if base_name not in self.base_models:
            logger.warning(f"Assigning node {node_id} to unknown base model '{base_name}'")

        if base_name not in self.node_assignments:
            self.node_assignments[base_name] = []
        if node_id not in self.node_assignments[base_name]:
            self.node_assignments[base_name].append(node_id)
            logger.info(f"Node {node_id} assigned to base model {base_name}")

    def remove_node_from_base_model(self, node_id: str, base_name: str):
        """Remove a node from a base model assignment."""
        if base_name in self.node_assignments:
            if node_id in self.node_assignments[base_name]:
                self.node_assignments[base_name].remove(node_id)

    def get_all_adapters(self) -> List[str]:
        """Get all adapter names across all base models."""
        return list(self._adapter_to_base.keys())

    def list_base_models(self) -> List[str]:
        """List all base model names."""
        return list(self.base_models.keys())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "base_models": {
                name: {
                    "name": cfg.name,
                    "model_id": cfg.model_id,
                    "adapters": cfg.adapters,
                    "adapter_paths": cfg.adapter_paths,
                    "gpu_memory_gb": cfg.gpu_memory_gb,
                    "max_workers": cfg.max_workers,
                }
                for name, cfg in self.base_models.items()
            },
            "node_assignments": self.node_assignments,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ClusterModelConfig":
        """Create from dictionary."""
        base_models = {}
        for name, cfg in data.get("base_models", {}).items():
            base_models[name] = BaseModelConfig(
                name=cfg["name"],
                model_id=cfg["model_id"],
                adapters=cfg.get("adapters", []),
                adapter_paths=cfg.get("adapter_paths", {}),
                gpu_memory_gb=cfg.get("gpu_memory_gb", 16.0),
                max_workers=cfg.get("max_workers", 32),
            )

        config = cls(base_models=base_models)
        config.node_assignments = data.get("node_assignments", {})
        return config

    @classmethod
    def from_yaml(cls, path: str) -> "ClusterModelConfig":
        """Load from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "ClusterModelConfig":
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)


def create_benchmark_config(
    num_base_models: int = 1,
    adapters_per_model: int = 50,
    base_model_ids: List[str] = None,
    adapter_prefix: str = "../sim-adapters/pool-10-r16/lora-",
) -> ClusterModelConfig:
    """
    Create a benchmark configuration with multiple base models and adapters.

    Adapter names use the format: {adapter_prefix}{N}
    Default: "sim-adapters/pool-10-r16/lora-0", "sim-adapters/pool-10-r16/lora-1", ...

    Args:
        num_base_models: Number of base models
        adapters_per_model: Number of adapters per base model
        base_model_ids: Optional list of HuggingFace model IDs
        adapter_prefix: Prefix for adapter names (default: "sim-adapters/pool-10-r16/lora-")

    Returns:
        ClusterModelConfig ready for benchmarking
    """
    if base_model_ids is None:
        from config import BASE_MODEL_ID
        base_model_ids = [
            BASE_MODEL_ID,
        ]
    else:
        base_model_ids = list(base_model_ids)  # Don't mutate caller's list

    # Ensure we have enough model IDs
    while len(base_model_ids) < num_base_models:
        base_model_ids.append(f"base-model-{len(base_model_ids)}")

    base_models = {}

    for i in range(num_base_models):
        model_id = base_model_ids[i]
        # Create short name from model ID
        short_name = model_id.split("/")[-1].lower().replace("-", "_")

        # Generate adapter names matching HF cache symlinks
        adapter_start = i * adapters_per_model
        adapters = [
            f"{adapter_prefix}{adapter_start + j}"
            for j in range(adapters_per_model)
        ]

        base_models[short_name] = BaseModelConfig(
            name=short_name,
            model_id=model_id,
            adapters=adapters,
        )

    return ClusterModelConfig(base_models=base_models)
