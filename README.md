# DeepQuant

A Python library for exporting Brevitas quantized neural networks.

## Installation

### Requirements

- Python 3.11 or higher
- PyTorch 2.1.2 or higher
- Brevitas 0.11.0 or higher

### Setup Environment

First, create and activate a new conda environment:

```bash
mamba create -n DeepQuant_env python=3.11
mamba activate DeepQuant_env
```

### Install Dependencies

Install PyTorch and its related packages:

```bash
mamba install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch
```

### Install the Package

Clone the repository and install in development mode:

```bash
cd DeepQuant
pip install -e .
```

## Running Tests

### Using Make (Recommended)

The project includes a Makefile with several testing commands:

```bash
# Run all tests with verbose output
make test

# Run only neural network test
make test-nn

# Run only multi-head attention test
make test-mha

# Run only CNN test
make test-cnn

# Run only Resnet18 test
make test-resnet

# Run a specific test file
make test-single TEST=test_simple_nn.py

# Show all available make commands
make help
```

### Using pytest directly

You can also run tests using pytest commands:

```bash
# Run all tests
python -m pytest src/DeepQuant/tests -v -s

# Run a specific test file
python -m pytest src/DeepQuant/tests/test_simple_nn.py -v -s
```

## Project Structure

```
DeepQuant/
├── Makefile
├── pyproject.toml
├── conftest.py
└── src/
    └── DeepQuant/
        ├── custom_forwards/
        │   ├── activations.py
        │   ├── linear.py
        │   └── multiheadattention.py
        ├── injects/
        │   ├── base.py
        │   ├── executor.py
        │   └── transformations.py
        ├── tests/
        │   ├── test_simple_mha.py
        │   ├── test_simple_nn.py
        │   └── test_simple_cnn.py
        ├── custom_tracer.py
        └── export_brevitas.py
```

### Key Components

- **Makefile**: Provides automation commands for testing
- **pyproject.toml**: Defines project metadata and dependencies for editable installation
- **conftest.py**: Pytest configuration file that handles warning filters

The source code is organized into several key modules:

- **custom_forwards/**: Contains the unrolled forward implementations for:

  - Linear layers (QuantLinear, QuantConv2d)
  - Activation functions (QuantReLU, QuantSigmoid, etc.)
  - Multi-head attention (QuantMultiheadAttention)

- **injects/**: Contains the transformation infrastructure:

  - Base transformation class and executor
  - Module-specific transformations
  - Validation and verification logic

- **tests/**: Example tests demonstrating the exporter usage:

  - Simple neural network (linear + activations)
  - Multi-head attention model
  - Convolutional neural network
  - Resnet18

- **custom_tracer.py**: Implements a specialized `CustomBrevitasTracer` for FX tracing

  - Handles Brevitas-specific module traversal
  - Ensures proper graph capture of quantization operations

- **export_brevitas.py**: Main API for end-to-end model export:
  - Orchestrates the transformation passes
  - Performs the final FX tracing
  - Validates model outputs through the process

## Usage

### Main Function: exportBrevitas

The main function of this library is `exportBrevitas`, which exports a Brevitas-based model to an FX GraphModule with unrolled quantization steps.

```python
from DeepQuant.export_brevitas import exportBrevitas

# Initialize your Brevitas model
model = YourBrevitasModel().eval()

# Create an input with the correct shape
input = torch.randn(1, input_channels, height, width)

# Export the model (with debug information)
fx_model = exportBrevitas(model, input, debug=True)
```

Arguments:

- `model`: The Brevitas-based model to export
- `example_input`: A representative input tensor for shape tracing
- `debug`: If True, prints transformation progress (default: False)

When `debug=True`, you'll see the output showing the progress, for example:

```
✓ MHA transformation successful - outputs match
✓ Linear transformation successful - outputs match
✓ Activation transformation successful - outputs match
All transformations completed successfully!
```

### Example Usage

A simple example script can be found in `example_usage.py` in the root directory of the project.

```python
import torch
import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8ActPerTensorFloat, Int32Bias
from DeepQuant.export_brevitas import exportBrevitas

# Define a simple quantized model
class SimpleQuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_quant = qnn.QuantIdentity(return_quant_tensor=True)
        self.conv = qnn.QuantConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            bias=True,
            weight_bit_width=4,
            bias_quant=Int32Bias,
            output_quant=Int8ActPerTensorFloat,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.conv(x)
        return x

# Export the model
model = SimpleQuantModel().eval()
dummy_input = torch.randn(1, 3, 32, 32)
fx_model = exportBrevitas(model, dummy_input, debug=True)
```
