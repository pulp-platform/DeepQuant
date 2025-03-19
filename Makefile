# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Federico Brancasi <fbrancasi@ethz.ch>

.PHONY: test test-nn test-mha test-cnn

# Test Directory
TEST_DIR = src/DeepQuant/tests

# Pytest flags
PYTEST_FLAGS = -v -s

# Target for running all tests
test:
	python -m pytest $(PYTEST_FLAGS) $(TEST_DIR)

# Target for running simple neural network test
test-nn:
	python -m pytest $(PYTEST_FLAGS) $(TEST_DIR)/test_simple_nn.py

# Target for running multi-head attention test
test-mha:
	python -m pytest $(PYTEST_FLAGS) $(TEST_DIR)/test_simple_mha.py

# Target for running convolutional neural network test
test-cnn:
	python -m pytest $(PYTEST_FLAGS) $(TEST_DIR)/test_simple_cnn.py

# Target for running resnet test
test-resnet:
	python -m pytest $(PYTEST_FLAGS) $(TEST_DIR)/test_resnet18.py

# Target for running mnist test
test-mnist:
	python -m pytest $(PYTEST_FLAGS) $(TEST_DIR)/test_mnist.py

# Target for running a specific test (usage: make test-single TEST=test_simple_nn.py)
test-single:
ifdef TEST
	python -m pytest $(PYTEST_FLAGS) $(TEST_DIR)/$(TEST)
else
	@echo "Please specify a test file with TEST=filename.py"
endif

# Show help
help:
	@echo "Available targets:"
	@echo "  make test      - Run all tests"
	@echo "  make test-nn   - Run simple neural network tests"
	@echo "  make test-mha  - Run multi-head attention tests"
	@echo "  make test-cnn  - Run convolutional neural network tests"
	@echo "  make test-single TEST=filename.py - Run a specific test file"