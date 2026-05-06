from typing import Callable, Dict


REGISTRY: Dict[str, Callable] = {}

from .extra_dists import EXTRA_DATASETS
REGISTRY.update(EXTRA_DATASETS)

def get_dataset_maker(name: str):
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {', '.join(sorted(REGISTRY))}"
        )
    return REGISTRY[name]

def list_datasets():
    return sorted(REGISTRY.keys())

__all__ = ["get_dataset_maker", "list_datasets", "REGISTRY"]
