#!/usr/bin/env python

try:
    import os
    import huggingface_hub as hf
except ImportError as e:
    print(f"utils/huggingface.py: Import Error: {e}")

# Download model checkpoint from HuggingFace repositories
def downloadFromHuggingFace(repo_id: str,
                            local_dir: str,
                            cache_dir: str,
                            apitoken: str,
                            revision: str = "main") -> str:
    
    # create cache dir if it does not exist
    os.makedirs(local_dir, exist_ok=True)

    # create download dir if does not exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # download checkpoing from huggingface..
    print(f"Downloading model checkpoint: {repo_id} to {local_dir}")
    # access HF to download files
    model_path = hf.snapshot_download(repo_id=repo_id,
                                      revision=revision, 
                                      token=apitoken,
                                      cache_dir=cache_dir,
                                      local_dir=local_dir)
    # done.
    print(f"Downloaded model checkpoint {model_path}")
    return model_path