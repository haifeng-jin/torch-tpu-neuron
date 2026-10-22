"""Integration and unit test suite for torch_tpu_neuron using pytest and torch.accelerator."""

import os
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tpu_neuron


@pytest.fixture(scope="session")
def neuron_backend():
  """Initializes the Neuron backend once for the test session."""
  info = torch_tpu_neuron.setup_neuron_backend()
  stats = torch_tpu_neuron.get_shim_stats()
  if stats["backend_mode"] == "uninitialized":
    pytest.skip(
        "AWS Trainium / Neuron runtime (libneuronxla.so) is not present on this host."
    )
  return {"info": info}


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

  stats = torch_tpu_neuron.get_shim_stats()
  assert stats["compile_count"] >= 1
  assert stats["execute_count"] >= 1
