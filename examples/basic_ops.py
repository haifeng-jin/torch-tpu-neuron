"""Basic tensor operations and accelerator inspection on AWS Trainium.

Usage:
    python examples/basic_ops.py
"""

# IMPORTANT: torch_tpu_neuron MUST be imported before torch
import torch_tpu_neuron
import torch

# 1. Inspect accelerator status
print(f"Accelerator available: {torch.accelerator.is_available()}")
print(f"Current accelerator:   {torch.accelerator.current_accelerator()}")
print(f"Device count:          {torch.accelerator.device_count()}")

props = torch.tpu.get_device_properties()
print(f"Device Name:           {props.name}")
print(f"Total Memory:          {props.total_memory / (1024**3):.1f} GiB")

# 2. Basic tensor operations on the accelerator
device = torch.accelerator.current_accelerator()

a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device=device)
c = a + b
print("\nElementwise addition (a + b):")
print(c.cpu())

# Matrix multiplication
x = torch.randn(4, 8, dtype=torch.float32, device=device)
w = torch.randn(8, 2, dtype=torch.float32, device=device)
y = torch.matmul(x, w)
print("\nMatrix multiplication (x @ w):")
print(y.cpu())

print("\nBasic ops verified successfully on AWS Trainium!")
