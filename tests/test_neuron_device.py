"""Integration test suite for torch_tpu_neuron on physical AWS Trainium / Neuron devices."""

import pytest
import torch_tpu_neuron
import torch
import torch.nn as nn
import torch.nn.functional as F


@pytest.fixture(scope="session")
def neuron_backend():
  """Initializes the Neuron backend once for the test session.

  Fails immediately if physical AWS Trainium / Neuron device or runtime is not present.
  """
  loaded = torch_tpu_neuron.setup_neuron_backend()
  assert loaded, (
      "Neuron backend failed to load. A physical AWS Trainium / Neuron device and "
      "runtime (libneuronpjrt.so, aws-neuronx-runtime-lib) are required."
  )
  assert torch_tpu_neuron.is_available(), "Neuron device is not available."
  assert torch.accelerator.is_available(), "torch.accelerator reports Neuron device is not available."
  assert torch.accelerator.device_count() >= 1, "No Neuron devices found on system."
  return {"loaded": True}


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

  bias_cpu = torch.tensor(1.5, dtype=torch.float32)
  bias_dev = bias_cpu.to(dev)

  c_cpu = torch.relu(torch.matmul(a_cpu, b_cpu)) + bias_cpu
  c_dev = torch.relu(torch.matmul(a_dev, b_dev)) + bias_dev
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


def test_sdp_kernel_disabling():
  """Verifies that precompiled TPU custom SDPA kernels are disabled via torch.backends.cuda."""
  if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
    if hasattr(torch.backends.cuda, "flash_sdp_enabled"):
      assert not torch.backends.cuda.flash_sdp_enabled()
    if hasattr(torch.backends.cuda, "mem_efficient_sdp_enabled"):
      assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    if hasattr(torch.backends.cuda, "math_sdp_enabled"):
      assert torch.backends.cuda.math_sdp_enabled()


def test_sdpa_attention_numerical_parity(neuron_backend):
  """Verifies scaled_dot_product_attention on Neuron device with seq_len >= 128 (pure StableHLO path)."""
  dev = torch.accelerator.current_accelerator()

  torch.manual_seed(99)
  batch, heads, seq_len, head_dim = 2, 4, 128, 64
  q_cpu = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
  k_cpu = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)
  v_cpu = torch.randn(batch, heads, seq_len, head_dim, dtype=torch.float32)

  out_cpu = F.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu, is_causal=True)

  q_dev = q_cpu.to(dev)
  k_dev = k_cpu.to(dev)
  v_dev = v_cpu.to(dev)
  out_dev = F.scaled_dot_product_attention(q_dev, k_dev, v_dev, is_causal=True)
  torch.accelerator.synchronize()

  assert torch.allclose(out_dev.cpu(), out_cpu, rtol=1e-3, atol=1e-3)
