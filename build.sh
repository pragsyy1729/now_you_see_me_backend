#!/bin/bash
# Install CPU-only PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Then install other requirements
pip install -r requirements.txt
