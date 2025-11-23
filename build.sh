#!/bin/bash
set -e
# Install regular dependencies first
pip install -r requirements.txt
# Install CPU-only PyTorch last (from PyTorch index)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
