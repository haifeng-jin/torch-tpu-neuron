import torch
import torch_tpu_neuron

# 1. Inspect accelerator status
print(f"Accelerator available: {torch.accelerator.is_available()}")
print(f"Current accelerator:   {torch.accelerator.current_accelerator()}")
print(f"Device count:          {torch.accelerator.device_count()}")

props = torch.tpu.get_device_properties()
print(f"Device Name:           {props.name}")  # "AWS Trainium (Neuron)"
print(f"Total Memory:          {props.total_memory / (1024**3):.1f} GiB")

# 2. Allocate and compute on the Neuron accelerator
device = torch.accelerator.current_accelerator()
x = torch.randn(32, 64, device=device)
y = torch.randn(64, 128, device=device)
z = torch.matmul(x, y)
print(f"Matmul output shape:   {z.shape}")


# 3. torch.compile with the accelerator backend
@torch.compile(backend="tpu")
def fn(a, b):
  return torch.relu(a @ b) + 1.0


out = fn(x, y)
print("Compiled execution succeeded!")
