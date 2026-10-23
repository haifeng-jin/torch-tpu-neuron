"""Basic operations, accelerator inspection, and MLP training on AWS Trainium.

Usage:
    python examples/basic_ops.py
"""

# IMPORTANT: torch_tpu_neuron MUST be imported before torch
import torch_tpu_neuron
import torch
import torch.nn as nn

# 1. Inspect accelerator status
print(f"Accelerator available: {torch.accelerator.is_available()}")
print(f"Current accelerator:   {torch.accelerator.current_accelerator()}")
print(f"Device count:          {torch.accelerator.device_count()}")

props = torch.tpu.get_device_properties()
print(f"Device Name:           {props.name}")
print(f"Total Memory:          {props.total_memory / (1024**3):.1f} GiB")

# 2. Setup synthetic non-linear regression problem on the accelerator
device = torch.accelerator.current_accelerator()
torch.manual_seed(42)

# Synthetic data: y = x @ w + 0.3 * sin(x_0)
x = torch.randn(256, 16, device=device)
w_true = torch.randn(16, 1, device=device)
y = torch.matmul(x, w_true) + 0.3 * torch.sin(x[:, :1])

# 3. Define a concise 2-layer regression MLP
model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)


# 4. Train with torch.compile using the Neuron accelerator backend
@torch.compile(backend="tpu")
def train_step(data: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  optimizer.zero_grad()
  pred = model(data)
  loss = criterion(pred, target)
  loss.backward()
  optimizer.step()
  return loss


print("\nTraining MLP regression model on synthetic data...")
initial_loss = 0.0
final_loss = 0.0

for step in range(1, 51):
  loss = train_step(x, y)
  loss_val = float(loss.item())
  if step == 1:
    initial_loss = loss_val
  if step % 10 == 0 or step == 1:
    print(f"  Step {step:02d} | Loss: {loss_val:.4f}")
  final_loss = loss_val

print(f"\nInitial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")
assert final_loss < initial_loss * 0.1, "Verification failed: loss did not decrease sufficiently!"
print("Training verified: loss decreased successfully on AWS Trainium!")
