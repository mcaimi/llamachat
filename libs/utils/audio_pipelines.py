#!/usr/bin/env python

try:
    import torch as t
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder
    import torchaudio.transforms as T
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Caught fatal exception: {e}")

# plot functions
def waveform(clip: AudioDecoder, title: str = "Waveform Plot") -> (plt.figure):
    # sample rate & channels
    sr = clip.metadata.sample_rate
    channels = clip.metadata.num_channels

    # get samples
    samples = clip.get_all_samples().data
    nc, ns = samples.shape

    # check
    if nc != channels:
        print(f"Mismatch: reported channels in metadata: {channels} differ from Tensor shape: {samples.shape}")
    
    # plot samples
    time_scale = t.arange(0, ns) / sr
    values = samples[0]
    
    fig, axis = plt.subplots(nc,1)
    axis.set_xlim([0,time_scale[-1]])
    axis.set_ylabel("Amplitude")
    axis.set_xlabel("Time (s)")
    axis.plot(time_scale, values)
    axis.set_title(title)

    # return
    return fig