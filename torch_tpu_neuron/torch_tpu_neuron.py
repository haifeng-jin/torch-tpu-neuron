"""TorchTPU Neuron Companion Library.

Enables running PyTorch on AWS Trainium / Neuron (trn1/trn2) via torch_tpu with
zero torch_tpu source code modifications, exposing the accelerator through standard
torch.accelerator APIs and the torch.neuron alias.
"""

from __future__ import annotations

import ctypes
import os
import sys

_INITIALIZED = False
_SHIM_SO: ctypes.CDLL | None = None


def _is_backend_loaded() -> bool:
  return bool(_SHIM_SO and hasattr(_SHIM_SO, "IsNeuronBackendLoaded") and _SHIM_SO.IsNeuronBackendLoaded())


def setup_neuron_backend() -> bool:
  """Initializes the Neuron PJRT backend and registers the device in PyTorch."""
  global _INITIALIZED, _SHIM_SO
  if _INITIALIZED:
    return _is_backend_loaded()

  shim_path = os.path.realpath(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtpu_neuron_shim.so")
  )
  if not os.path.exists(shim_path):
    raise FileNotFoundError(f"Missing {shim_path}. Build extension with `pip install -e .`")

  if not os.environ.get("TORCH_TPU_NEURON_BACKEND_LIB"):
    try:
      import libneuronxla  # type: ignore

      lib_path = getattr(libneuronxla, "get_library_path", lambda: None)()
      if lib_path:
        os.environ["TORCH_TPU_NEURON_BACKEND_LIB"] = lib_path
    except Exception:
      pass

  os.environ["TPU_LIBRARY_PATH"] = shim_path
  nodelete = getattr(os, "RTLD_NODELETE", 0x1000)
  _SHIM_SO = ctypes.CDLL(shim_path, mode=os.RTLD_NOW | os.RTLD_LOCAL | nodelete)
  _SHIM_SO.IsNeuronBackendLoaded.restype = ctypes.c_int

  if not _is_backend_loaded():
    return False

  from torch_tpu._internal.utils import annotations, hardware

  annotations._internal_apis_allowed_for_testing = True

  def _device_count() -> int:
    if not _is_backend_loaded():
      return 0
    try:
      from torch_tpu._internal.device import _device_ops_backend

      return _device_ops_backend._get_device_count()
    except Exception:
      return 0

  neuron_spec = hardware.TpuSpec(
      name="AWS Trainium (Neuron)",
      devices_per_chip=1,
      hbm_bytes_per_chip=32 * 1024**3,
      bf16_flops_per_chip=190 * 10**12,
      topologies={
          1: (1, 1, 1),
          2: (2, 1, 1),
          4: (2, 2, 1),
          8: (2, 4, 1),
          16: (4, 4, 1),
          32: (4, 8, 1),
      },
      sparse_core_spec=None,
  )
  hardware.get_tpu_spec = lambda *a, **kw: neuron_spec
  hardware.get_tpu_device_name = lambda *a, **kw: neuron_spec.name
  hardware.get_hbm_bytes_per_device = lambda *a, **kw: 32 * 1024**3
  hardware.get_hbm_bytes_per_chip = lambda *a, **kw: 32 * 1024**3
  hardware.get_devices_per_chip = lambda *a, **kw: 1
  hardware.get_bf16_flops_per_chip = lambda *a, **kw: 190 * 10**12
  hardware.get_bf16_flops_per_device = lambda *a, **kw: 190 * 10**12
  hardware.get_tpu_device_count = lambda *a, **kw: _device_count()
  hardware._scan_pci_tpus = lambda *a, **kw: (_device_count(), neuron_spec)

  import torch
  from torch_tpu import _loader

  _loader._init_device("tpu")
  _loader._LOADED = True
  _device_count()

  if hasattr(torch, "tpu"):
    setattr(torch, "neuron", torch.tpu)
    sys.modules["torch.neuron"] = sys.modules["torch.tpu"]

  _INITIALIZED = True
  return True


def is_available() -> bool:
  """Returns True if the Neuron accelerator backend is available."""
  import torch

  if not setup_neuron_backend():
    return False
  if hasattr(torch, "accelerator"):
    return bool(torch.accelerator.is_available())
  return bool(getattr(torch, "tpu", None) and torch.tpu.is_available())
