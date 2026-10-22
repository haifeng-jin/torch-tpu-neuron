"""Integration and unit test suite for torch_tpu_neuron using pytest and torch.accelerator."""

import ctypes
import os
import shutil
import subprocess
import pytest
import torch_tpu_neuron
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.fixture(scope="session")
def neuron_backend(tmp_path_factory):
  """Initializes the Neuron backend once for the test session.

  If no physical AWS Trainium runtime is available, compiles and loads a dummy
  libneuronxla.so to enable testing device registration and option filtering.
  """
  loaded = torch_tpu_neuron.setup_neuron_backend()
  if loaded:
    return {"loaded": True, "is_dummy": False, "lib_path": None}

  # Build dummy libneuronxla.so
  dummy_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy_neuronxla.cc")
  dummy_dir = tmp_path_factory.mktemp("dummy_neuron")
  dummy_so = os.path.join(dummy_dir, "libneuronxla.so")

  cxx = os.environ.get("CXX", "g++")
  cmd = [cxx, "-O2", "-shared", "-fPIC", "-std=c++17", dummy_src, "-o", dummy_so]
  subprocess.check_call(cmd)

  os.environ["TORCH_TPU_NEURON_BACKEND_LIB"] = dummy_so
  loaded = torch_tpu_neuron.setup_neuron_backend()
  if not loaded:
    pytest.skip("Failed to initialize dummy Neuron backend.")

  return {"loaded": True, "is_dummy": True, "lib_path": dummy_so}


def test_dummy_rejects_premapped_buffer_size_without_shim(neuron_backend):
  """Verifies dummy_neuronxla rejects premapped_buffer_size when called directly without shim."""
  if not neuron_backend.get("is_dummy"):
    pytest.skip("Test requires dummy neuron PJRT plugin.")

  lib_path = neuron_backend["lib_path"]
  lib = ctypes.CDLL(lib_path)

  class PJRT_NamedValue(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("name", ctypes.c_char_p),
        ("name_size", ctypes.c_size_t),
        ("type", ctypes.c_int),
        ("int64_value", ctypes.c_int64),
        ("value_size", ctypes.c_size_t),
    ]

  class PJRT_Client_Create_Args(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("create_options", ctypes.POINTER(PJRT_NamedValue)),
        ("num_options", ctypes.c_size_t),
        ("kv_get_callback", ctypes.c_void_p),
        ("kv_get_user_arg", ctypes.c_void_p),
        ("kv_put_callback", ctypes.c_void_p),
        ("kv_put_user_arg", ctypes.c_void_p),
        ("client", ctypes.c_void_p),
        ("kv_try_get_callback", ctypes.c_void_p),
        ("kv_try_get_user_arg", ctypes.c_void_p),
    ]

  class PJRT_Api(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("extension_start", ctypes.c_void_p),
        ("version_struct_size", ctypes.c_size_t),
        ("version_ext", ctypes.c_void_p),
        ("major_version", ctypes.c_int),
        ("minor_version", ctypes.c_int),
        ("methods_before", ctypes.c_void_p * 10),
        ("PJRT_Client_Create", ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.POINTER(PJRT_Client_Create_Args))),
    ]

  lib.GetPjrtApi.restype = ctypes.POINTER(PJRT_Api)
  api = lib.GetPjrtApi().contents

  opt_name = b"premapped_buffer_size"
  opt = PJRT_NamedValue(
      struct_size=ctypes.sizeof(PJRT_NamedValue),
      extension_start=None,
      name=opt_name,
      name_size=len(opt_name),
      type=1,
      int64_value=1048576,
      value_size=1,
  )
  args = PJRT_Client_Create_Args(
      struct_size=ctypes.sizeof(PJRT_Client_Create_Args),
      extension_start=None,
      create_options=ctypes.pointer(opt),
      num_options=1,
      client=None,
  )

  err = api.PJRT_Client_Create(ctypes.byref(args))
  assert err is not None, "dummy_neuronxla must reject premapped_buffer_size with an error"


def test_device_registration_and_metadata(neuron_backend):
  """Verifies device registration, availability, and accelerator metadata."""
  assert torch_tpu_neuron.is_available()
  assert torch.accelerator.is_available()
  assert hasattr(torch, "neuron")
  assert getattr(torch, "neuron").is_available()

  assert torch.accelerator.device_count() >= 1

  props = torch.tpu.get_device_properties()
  assert props.name == "AWS Trainium (Neuron)"
  assert props.total_memory == 32 * 1024**3
  if hasattr(props, "devices_per_chip"):
    assert props.devices_per_chip == 1

  acc = torch.accelerator.current_accelerator()
  assert acc is not None


def test_eager_tensor_ops_numerical_parity(neuron_backend):
  """Verifies eager arithmetic operations on the Neuron accelerator against CPU ground truth."""
  if neuron_backend.get("is_dummy"):
    pytest.skip("Skipping hardware execution test on dummy PJRT plugin.")

  dev = torch.accelerator.current_accelerator()

  torch.manual_seed(123)
  a_cpu = torch.randn(16, 32, dtype=torch.float32)
  b_cpu = torch.randn(32, 64, dtype=torch.float32)

  a_dev = a_cpu.to(dev)
  b_dev = b_cpu.to(dev)

  c_cpu = torch.relu(torch.matmul(a_cpu, b_cpu)) + 1.5
  c_dev = torch.relu(torch.matmul(a_dev, b_dev)) + 1.5
  torch.accelerator.synchronize()

  assert c_dev.device.type == dev.type
  assert torch.allclose(c_dev.cpu(), c_cpu, rtol=1e-4, atol=1e-4)


def test_autograd_forward_backward(neuron_backend):
  """Verifies autograd forward and backward pass on an MLP on the Neuron accelerator."""
  if neuron_backend.get("is_dummy"):
    pytest.skip("Skipping hardware execution test on dummy PJRT plugin.")

  dev = torch.accelerator.current_accelerator()

  class SimpleMLP(nn.Module):

    def __init__(self):
      super().__init__()
      self.fc1 = nn.Linear(16, 32)
      self.fc2 = nn.Linear(32, 8)

    def forward(self, x):
      return self.fc2(F.gelu(self.fc1(x)))

  torch.manual_seed(42)
  model_cpu = SimpleMLP()
  x_cpu = torch.randn(4, 16, requires_grad=True)

  model_dev = SimpleMLP().to(dev)
  model_dev.load_state_dict(model_cpu.state_dict())
  x_dev = x_cpu.detach().clone().to(dev).requires_grad_(True)

  out_cpu = model_cpu(x_cpu)
  out_dev = model_dev(x_dev)
  torch.accelerator.synchronize()
  assert torch.allclose(out_dev.cpu(), out_cpu, rtol=1e-4, atol=1e-4)

  loss_cpu = out_cpu.sum()
  loss_dev = out_dev.sum()
  loss_cpu.backward()
  loss_dev.backward()
  torch.accelerator.synchronize()

  assert x_dev.grad is not None
  assert torch.allclose(x_dev.grad.cpu(), x_cpu.grad, rtol=1e-4, atol=1e-4)

  for (n_cpu, p_cpu), (n_dev, p_dev) in zip(
      model_cpu.named_parameters(), model_dev.named_parameters()
  ):
    assert p_dev.grad is not None
    assert torch.allclose(p_dev.grad.cpu(), p_cpu.grad, rtol=1e-4, atol=1e-4)


def test_torch_compile(neuron_backend):
  """Verifies torch.compile with Neuron accelerator backend."""
  if neuron_backend.get("is_dummy"):
    pytest.skip("Skipping hardware execution test on dummy PJRT plugin.")

  dev = torch.accelerator.current_accelerator()

  def my_fn(x, y):
    return torch.sin(x) * torch.cos(y) + 2.0

  compiled_fn = torch.compile(my_fn, backend="tpu")

  x_cpu = torch.randn(8, 8, dtype=torch.float32)
  y_cpu = torch.randn(8, 8, dtype=torch.float32)
  expected_cpu = my_fn(x_cpu, y_cpu)

  x_dev = x_cpu.to(dev)
  y_dev = y_cpu.to(dev)

  actual_dev = compiled_fn(x_dev, y_dev)
  torch.accelerator.synchronize()
  assert torch.allclose(actual_dev.cpu(), expected_cpu, rtol=1e-4, atol=1e-4)
