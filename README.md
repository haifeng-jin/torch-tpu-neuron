# torch-tpu-neuron

**Companion library enabling Google `torch_tpu` to seamlessly target Amazon Trainium / Neuron (`trn1`/`trn2`) with ZERO modifications to `torch_tpu` code.**

---

## Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/haifeng-jin/torch-tpu-neuron.git
cd torch-tpu-neuron

# Build and install in editable mode
pip install -e .
```

### Quickstart

> **Note**: `torch_tpu_neuron` must be imported **before** `torch` so that PyTorch's backend loader registers the Neuron PJRT shim runtime.

```python
import torch_tpu_neuron  # Must be imported before torch
import torch

device = torch.accelerator.current_accelerator()

a = torch.tensor([1.0, 2.0, 3.0], device=device)
b = torch.tensor([4.0, 5.0, 6.0], device=device)
print(a + b)
```

See [`example.py`](example.py) for a full example of accelerator inspection and `@torch.compile` execution.

---

## Environment Variables (Optional)

**No environment variables are required.** `import torch_tpu_neuron` automatically discovers `libneuronxla.so` and configures the runtime without any manual setup.

For custom library paths, the following optional override is supported:

| Variable | Description |
|---|---|
| `TORCH_TPU_NEURON_BACKEND_LIB` | Override path to Neuron PJRT runtime `.so` (defaults to auto-detecting `libneuronxla.so`) |

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
