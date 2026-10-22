"""TorchTPU Neuron Companion Library.

Enables running PyTorch on AWS Trainium / Neuron (trn1/trn2) via torch_tpu with
zero torch_tpu source code modifications, exposing the accelerator through standard
torch.accelerator APIs and the torch.neuron alias.
"""

from __future__ import annotations

import ctypes
import glob
import os
import sys
from typing import Any

_PROT_READ = 0x1
_PROT_WRITE = 0x2
_PROT_EXEC = 0x4
_PROT_RWX = _PROT_READ | _PROT_WRITE | _PROT_EXEC
_PROT_RX = _PROT_READ | _PROT_EXEC

_SHIM_INITIALIZED: bool = False
_SHIM_SO: ctypes.CDLL | None = None


def find_library_file(filename: str) -> str | None:
  """Locates a shared library in package, virtualenv, system, or library paths."""
  cur_dir = os.path.dirname(os.path.abspath(__file__))
  candidates = [
      os.path.join(cur_dir, filename),
      os.path.join(cur_dir, "csrc", filename),
      os.path.join(cur_dir, "..", "csrc", filename),
      os.path.join(cur_dir, "..", filename),
      os.path.join(sys.prefix, "lib", filename),
      os.path.join(sys.prefix, "lib64", filename),
      os.path.abspath(filename),
      f"/opt/aws/neuron/lib/{filename}",
      f"/usr/local/lib/{filename}",
      f"/usr/lib/{filename}",
  ]

  # Search sys.path and subpackages (including installed site-packages).
  for path in sys.path:
    candidates.append(os.path.join(path, "torch_tpu_neuron", filename))
    candidates.append(os.path.join(path, "libneuronxla", filename))
    candidates.append(os.path.join(path, "torch_tpu", filename))
    candidates.append(os.path.join(path, filename))

  # Check standard library search paths from environment.
  for env_dir_var in ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"]:
    env_val = os.environ.get(env_dir_var)
    if env_val:
      for part in env_val.split(os.pathsep):
        if not part:
          continue
        direct = os.path.join(part, filename)
        if os.path.exists(direct):
          return os.path.abspath(direct)
        candidates.append(direct)

  # Check exact candidates
  for cand in candidates:
    if os.path.exists(cand):
      return os.path.abspath(cand)

  # Check glob prefix for version-tagged .so files
  if filename.endswith(".so"):
    stem = filename[:-3]
    for search_dir in [cur_dir, os.path.join(cur_dir, "..")]:
      for match in glob.glob(os.path.join(search_dir, f"{stem}*.so")):
        if os.path.exists(match):
          return os.path.abspath(match)
    for path in sys.path:
      for match in glob.glob(os.path.join(path, "torch_tpu_neuron", f"{stem}*.so")):
        if os.path.exists(match):
          return os.path.abspath(match)
      for match in glob.glob(os.path.join(path, "libneuronxla", f"{stem}*.so")):
        if os.path.exists(match):
          return os.path.abspath(match)

  # Check system dynamic linker fallback.
  try:
    handle = ctypes.CDLL(filename)
    if handle:
      return filename
  except Exception:
    pass

  return None


def setup_neuron_backend() -> dict[str, Any]:
  """Initializes the Neuron PJRT backend and registers the device in PyTorch.

  Returns:
    Metadata dictionary describing the initialized backend.
  """
  global _SHIM_INITIALIZED, _SHIM_SO
  if _SHIM_INITIALIZED:
    stats = get_shim_stats()
    if stats["backend_mode"] != "uninitialized":
      return stats

  # 1. Locate libraries.
  shim_path = find_library_file("libtpu_neuron_shim.so")
  if not shim_path:
    raise FileNotFoundError(
        "Could not locate 'libtpu_neuron_shim.so'. "
        "Build the C++ extension first with `pip install -e .`"
    )

  # Discover real Trainium / Neuron runtime library (libneuronxla.so).
  neuron_lib = os.environ.get("TORCH_TPU_NEURON_BACKEND_LIB")
  if not neuron_lib:
    try:
      import libneuronxla  # type: ignore
      neuron_lib = getattr(libneuronxla, "get_library_path", lambda: None)()
    except Exception:
      pass
  if not neuron_lib:
    neuron_lib = find_library_file("libneuronxla.so")
  if neuron_lib:
    os.environ["TORCH_TPU_NEURON_BACKEND_LIB"] = neuron_lib

  prev_tpu_lib = os.environ.get("TPU_LIBRARY_PATH")
  libtpu_candidate = None
  try:
    import libtpu
    libtpu_candidate = libtpu.get_library_path()
  except Exception:
    pass

  # 2. Configure environment variables for torch_tpu and PJRT.
  os.environ["TPU_LIBRARY_PATH"] = shim_path

  plugin_entry = f"tpu:{shim_path},neuron:{shim_path}"
  existing_pjrt = os.environ.get("PJRT_NAMES_AND_LIBRARY_PATHS", "")
  if existing_pjrt:
    if "tpu:" not in existing_pjrt:
      os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = f"{existing_pjrt},{plugin_entry}"
  else:
    os.environ["PJRT_NAMES_AND_LIBRARY_PATHS"] = plugin_entry

  # 3. Load shim and apply in-process PJRT_Api table redirection.
  nodelete_flag = getattr(os, "RTLD_NODELETE", 0x1000)
  rtld_mode = os.RTLD_NOW | os.RTLD_LOCAL | nodelete_flag

  try:
    _SHIM_SO = ctypes.CDLL(shim_path, mode=rtld_mode)
    _SHIM_SO.GetPjrtApi.restype = ctypes.c_void_p
    shim_api_ptr = _SHIM_SO.GetPjrtApi()

    # Targets to redirect: only previously-loaded libtpu.so candidates (not global or real neuron)
    target_ptrs = []
    excluded_paths = {
        os.path.abspath(p)
        for p in [
            shim_path,
            os.environ.get("TORCH_TPU_NEURON_BACKEND_LIB"),
        ]
        if p
    }

    for cand_path in [prev_tpu_lib, libtpu_candidate]:
      if cand_path and os.path.exists(cand_path):
        abs_cand = os.path.abspath(cand_path)
        if abs_cand in excluded_paths:
          continue
        try:
          cand_cdll = ctypes.CDLL(cand_path)
          if hasattr(cand_cdll, "GetPjrtApi"):
            cand_cdll.GetPjrtApi.restype = ctypes.c_void_p
            cand_ptr = cand_cdll.GetPjrtApi()
            if cand_ptr and cand_ptr != shim_api_ptr:
              target_ptrs.append(cand_ptr)
        except Exception:
          pass

    pagesize = os.sysconf("SC_PAGE_SIZE")
    page_mask = ~(pagesize - 1)
    libc = ctypes.cdll.LoadLibrary("libc.so.6")

    for tpu_api_ptr in target_ptrs:
      if tpu_api_ptr and shim_api_ptr and tpu_api_ptr != shim_api_ptr:
        tpu_struct_size = ctypes.c_size_t.from_address(tpu_api_ptr).value
        shim_struct_size = ctypes.c_size_t.from_address(shim_api_ptr).value
        copy_size = min(tpu_struct_size, shim_struct_size)

        page_start = tpu_api_ptr & page_mask
        end_addr = tpu_api_ptr + copy_size
        length = (end_addr - page_start + pagesize - 1) & page_mask

        res = libc.mprotect(
            ctypes.c_void_p(page_start), ctypes.c_size_t(length), _PROT_READ | _PROT_WRITE
        )
        if res == 0:
          ctypes.memmove(tpu_api_ptr, shim_api_ptr, copy_size)
  except Exception as e:
    sys.stderr.write(f"[torch_tpu_neuron] Warning during PJRT table redirection: {e}\n")

  # 4. Patch torch_tpu hardware discovery and annotations without touching torch_tpu source files.
  try:
    from torch_tpu._internal.utils import annotations
    annotations._internal_apis_allowed_for_testing = True
  except Exception:
    pass

  try:
    from torch_tpu._internal.utils import hardware

    def _get_effective_device_count() -> int:
      stats = get_shim_stats()
      if stats["backend_mode"] != "uninitialized":
        try:
          from torch_tpu._internal.device import _device_ops_backend
          return _device_ops_backend._get_device_count()
        except Exception:
          pass
      return 0

    hardware.get_tpu_device_name = lambda *args, **kwargs: "AWS Trainium (Neuron)"
    hardware.get_hbm_bytes_per_device = lambda *args, **kwargs: 32 * 1024**3
    hardware.get_hbm_bytes_per_chip = lambda *args, **kwargs: 32 * 1024**3
    hardware.get_devices_per_chip = lambda *args, **kwargs: 1
    hardware.get_bf16_flops_per_chip = lambda *args, **kwargs: 190 * 10**12
    hardware.get_bf16_flops_per_device = lambda *args, **kwargs: 190 * 10**12
    hardware.get_tpu_device_count = lambda *args, **kwargs: _get_effective_device_count()

    if hasattr(hardware, "TpuSpec"):
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
      hardware.get_tpu_spec = lambda *args, **kwargs: neuron_spec
      hardware._scan_pci_tpus = lambda *args, **kwargs: (_get_effective_device_count(), neuron_spec)
    else:
      hardware._scan_pci_tpus = lambda *args, **kwargs: (
          _get_effective_device_count(),
          {1: "1,1,1", 2: "2,1,1", 4: "2,2,1", 8: "2,4,1", 16: "4,4,1", 32: "4,8,1"},
      )
  except Exception as e:
    sys.stderr.write(f"[torch_tpu_neuron] Warning patching hardware: {e}\n")

  # 5. Initialize torch_tpu device and register 'torch.neuron' alias.
  import torch
  from torch_tpu import _loader

  _loader._init_device("tpu")
  _loader._LOADED = True

  if hasattr(torch, "tpu"):
    setattr(torch, "neuron", torch.tpu)
    sys.modules["torch.neuron"] = sys.modules["torch.tpu"]

  # Eagerly initialize the PJRT backend to ensure deviceCount() is accurate.
  if hasattr(torch, "accelerator"):
    torch.accelerator.is_available()

  stats = get_shim_stats()
  if stats["backend_mode"] != "uninitialized":
    _SHIM_INITIALIZED = True
  return stats


def get_shim_stats() -> dict[str, Any]:
  """Returns telemetry and runtime stats from the Neuron shim."""
  global _SHIM_SO
  if _SHIM_SO is None:
    shim_path = find_library_file("libtpu_neuron_shim.so")
    if shim_path:
      _SHIM_SO = ctypes.CDLL(shim_path, mode=ctypes.RTLD_GLOBAL)

  if _SHIM_SO and hasattr(_SHIM_SO, "GetNeuronShimStats"):
    compile_count = ctypes.c_int64(0)
    execute_count = ctypes.c_int64(0)
    last_program_size = ctypes.c_int64(0)
    backend_mode = ctypes.c_char_p()

    _SHIM_SO.GetNeuronShimStats(
        ctypes.byref(compile_count),
        ctypes.byref(execute_count),
        ctypes.byref(last_program_size),
        ctypes.byref(backend_mode),
    )
    mode_str = backend_mode.value.decode("utf-8") if backend_mode.value else "unknown"
    return {
        "compile_count": compile_count.value,
        "execute_count": execute_count.value,
        "last_program_size": last_program_size.value,
        "backend_mode": mode_str,
    }
  return {
      "compile_count": 0,
      "execute_count": 0,
      "last_program_size": 0,
      "backend_mode": "uninitialized",
  }


def device(index: int | None = None) -> Any:
  """Returns a torch.device pointing to the Neuron accelerator device."""
  import torch
  setup_neuron_backend()
  if hasattr(torch, "accelerator") and torch.accelerator.is_available():
    acc = torch.accelerator.current_accelerator()
    if index is not None:
      return torch.device(acc.type, index)
    return acc
  return torch.device("tpu", index)


def is_available() -> bool:
  """Returns True if the Neuron accelerator backend is available."""
  import torch
  setup_neuron_backend()
  stats = get_shim_stats()
  if stats["backend_mode"] == "uninitialized":
    return False
  if hasattr(torch, "accelerator"):
    return bool(torch.accelerator.is_available())
  return bool(getattr(torch, "tpu", None) and torch.tpu.is_available())
