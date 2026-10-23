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


def _disable_tpu_custom_sdp_kernels() -> None:
  """Disables TPU precompiled FlashAttention kernels via torch.backends.cuda.

  Forces torch_tpu to lower F.scaled_dot_product_attention to pure StableHLO
  (ScaledDotProductFusedAttentionShlo) instead of @tpu_custom_call Mosaic kernels.
  """
  try:
    import torch

    if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
      if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(False)
      if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(False)
      if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
      if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)
  except Exception:
    pass


def setup_neuron_backend() -> bool:
  """Initializes the Neuron PJRT backend and registers the device in PyTorch."""
  global _INITIALIZED, _SHIM_SO
  if _INITIALIZED:
    return _is_backend_loaded()

  _disable_tpu_custom_sdp_kernels()

  shim_path = os.path.realpath(
      os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtpu_neuron_shim.so")
  )
  if not os.path.exists(shim_path):
    raise FileNotFoundError(f"Missing {shim_path}. Build extension with `pip install -e .`")

  if not os.environ.get("TORCH_TPU_NEURON_BACKEND_LIB"):
    try:
      import libneuronxla  # type: ignore

      if hasattr(libneuronxla, "libneuronpjrt_path"):
        os.environ["TORCH_TPU_NEURON_BACKEND_LIB"] = libneuronxla.libneuronpjrt_path.libneuronpjrt_path()
      elif hasattr(libneuronxla, "get_library_path"):
        os.environ["TORCH_TPU_NEURON_BACKEND_LIB"] = libneuronxla.get_library_path()
      else:
        candidate = os.path.join(os.path.dirname(libneuronxla.__file__), "libneuronpjrt.so")
        if os.path.exists(candidate):
          os.environ["TORCH_TPU_NEURON_BACKEND_LIB"] = candidate
    except Exception:
      pass

  # Preload libnrt into the global symbol table so libneuronpjrt can resolve NRT symbols
  for nrt_cand in ["/opt/aws/neuron/lib/libnrt.so.1", "/usr/local/lib/libnrt.so.1", "libnrt.so.1"]:
    try:
      ctypes.CDLL(nrt_cand, mode=ctypes.RTLD_GLOBAL)
      break
    except Exception:
      pass

  # Ensure neuronxcc.nki alias is present if nki is installed
  try:
    import nki
    import neuronxcc

    sys.modules.setdefault("neuronxcc.nki", nki)
    if not hasattr(neuronxcc, "nki"):
      neuronxcc.nki = nki
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
      return 1

  device_name = "AWS Trainium (Neuron)"
  memory_bytes = 32 * 1024**3
  bf16_flops = 190 * 10**12

  hardware.get_tpu_device_name = lambda *a, **kw: device_name
  hardware.get_hbm_bytes_per_device = lambda *a, **kw: memory_bytes
  hardware.get_hbm_bytes_per_chip = lambda *a, **kw: memory_bytes
  hardware.get_devices_per_chip = lambda *a, **kw: 1
  hardware.get_bf16_flops_per_chip = lambda *a, **kw: bf16_flops
  hardware.get_bf16_flops_per_device = lambda *a, **kw: bf16_flops
  hardware.get_tpu_device_count = lambda *a, **kw: _device_count()
  hardware._scan_pci_tpus = lambda *a, **kw: (_device_count(), {1: "1,1,1", 2: "1,2,1"})

  import torch
  from torch_tpu import _loader

  _loader._init_device("tpu")
  _loader._LOADED = True
  _device_count()

  if hasattr(torch, "tpu"):
    setattr(torch, "neuron", torch.tpu)
    sys.modules["torch.neuron"] = sys.modules["torch.tpu"]

  if hasattr(torch, "accelerator"):
    torch.accelerator.current_accelerator = lambda check_available=False: torch.device("tpu")
    torch.accelerator.is_available = lambda: bool(getattr(torch, "tpu", None) and torch.tpu.is_available())
    torch.accelerator.device_count = lambda: int(getattr(torch, "tpu", None) and torch.tpu.device_count() or 0)

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


_disable_tpu_custom_sdp_kernels()
