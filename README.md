# DeepQuant

A library for true-quantization and optimization of neural networks.

Deeploy is developed as part of the PULP project, a joint effort between ETH Zurich and the University of Bologna.

## License

Unless specified otherwise in the respective file headers, all code checked into this repository is made available under a permissive license. All software sources and tool scripts are licensed under Apache 2.0, except for files contained in the `scripts` directory, which are licensed under the MIT license, and files contained in the `DeeployTest/Tests`directory, which are licensed under the [Creative Commons Attribution-NoDerivates 4.0 International](https://creativecommons.org/licenses/by-nd/4.0) license (CC BY-ND 4.0).

## Installation

Start by creating a new env with `Python 3.11` or higher. Then clone the repo and install the library as an editable package with:
```
pip install -e .
```

## Running Tests

We provide comprehensive tests with pytest, to execute all tests, simply run `pytest`. We mark our tests in two categories, `SingleLayerTests` and `ModelTests`, to execute the tests of the specific category, you can run `pytest -m <category>`. For instance, to execute only the single layer tests, you can run `pytest -m SingleLayerTests`.

## ⚠️ Disclaimer ⚠️
This library is currently in **beta stage** and under active development. Interfaces and features are subject to change, and stability is not yet guaranteed. Use at your own risk, and feel free to report any issues or contribute to its improvement.