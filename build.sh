#!/bin/bash
set -e
# Install regular dependencies first
pip install -r requirements.txt
# Install CPU-only PyTorch last (from PyTorch index)
pip install torch torchvision