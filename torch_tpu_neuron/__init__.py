"""torch_tpu_neuron: Companion library for running torch_tpu on AWS Trainium / Neuron."""

from torch_tpu_neuron.torch_tpu_neuron import (
    setup_neuron_backend,
    is_available,
)

__all__ = [
    "setup_neuron_backend",
    "is_available",
]

# Automatically initialize backend on import if Neuron runtime is available.
try:
  setup_neuron_backend()
except Exception:
  pass
