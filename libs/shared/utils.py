#!/usr/bin/env python

try:
    import torch
except ImportError as e:
    print(f"Caught fatal exception: {e}")

# detect accelerator
def detect_accelerator() -> (str, torch.dtype):
    # detect discrete accelerator
    if torch.cuda.is_available():
        accelerator = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        dtype = torch.float32
    else:
        accelerator = "cpu"
        dtype = torch.float32

    return (accelerator, dtype)
