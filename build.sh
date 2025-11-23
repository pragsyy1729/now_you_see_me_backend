#!/bin/bash
set -e
# Install regular dependencies first
pip install -r requirements.txt
# Install CPU-only PyTorch last (from PyTorch index)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
