# torch-tpu-neuron

**A prototype library enabling Google `torch_tpu` to target Amazon Trainium / Neuron (`trn1`/`trn2`) without modifying `torch_tpu` code.**

---

## Prerequisites

* Physical AWS Trainium instance (`trn1` / `trn2`).
* Neuron driver and runtime (`aws-neuronx-runtime-lib >= 2.35`).

---

## Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/haifeng-jin/torch-tpu-neuron.git
cd torch-tpu-neuron

# Install in editable mode with the AWS Neuron package index
pip install -e . --extra-index-url=https://pip.repos.neuron.amazonaws.com
```

### Quickstart

> **Important**: `torch_tpu_neuron` must be imported **first**, before `torch` or any torch-dependent library, so that PyTorch registers the Neuron PJRT shim runtime.

```python
import torch_tpu_neuron  # Must be imported before torch
import torch

device = torch.accelerator.current_accelerator()

a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32, device=device)
print(a + b)
```

---

## Examples

All examples are organized under the [`examples/`](examples/) directory:

### Device Inspection & Regression MLP Training (`examples/basic_ops.py`)

Inspects the accelerator, generates synthetic non-linear regression data on the Neuron device, and trains a 2-layer MLP using `torch.compile(backend="tpu")`, verifying that loss decreases and converges:

```bash
python examples/basic_ops.py
```

---

## Known Limitations

1. **No Float64 (`f64`) Support**: Trainium hardware does not support 64-bit float precision. Mixing Python scalar floats with tensors (e.g., `0.3 * x` or `x + 1e-6`) creates `f64` constants in the graph that the Neuron compiler will reject (`[NCC_ESPP004]`). Always use explicit `float32` or `bfloat16` tensors for constants.
2. **`torch.compile` Scope**: Compile the model directly (`model = torch.compile(raw_model, backend="tpu")`) and run the optimizer in eager mode. Wrapping the whole training step with optimizers like Adam captures CPU step counter tensors into the TPU graph, triggering device placement errors.

---

## Directory Structure

```text
torch-tpu-neuron/
├── csrc/
│   └── libtpu_neuron_shim.cc       # PJRT C API shim mapping libtpu to libneuronpjrt.so
├── torch_tpu_neuron/
│   ├── __init__.py
│   └── torch_tpu_neuron.py         # Runtime setup & torch.backends.cuda SDP configuration
├── examples/
│   └── basic_ops.py                # Device inspection, torch.compile & MLP regression training
├── tests/
│   └── test_neuron_device.py       # Integration test suite
├── pyproject.toml
└── README.md
```

---

## Environment Variables (Optional)

**No environment variables are required.** `import torch_tpu_neuron` automatically discovers `libneuronpjrt.so` and configures the runtime.

For custom library paths, the following optional override is supported:

| Variable | Description |
|---|---|
| `TORCH_TPU_NEURON_BACKEND_LIB` | Override path to Neuron PJRT runtime `.so` (defaults to auto-detecting `libneuronpjrt.so`) |

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
