#!/usr/bin/env python

try:
    from io import BytesIO
    import torch as t
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder
    import torchaudio.transforms as T
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    print(f"Caught fatal exception: {e}")

FREQUENCY_BINS = 512

# resample a clip
def resample(audioClip: AudioDecoder, target_sample_rate: int = 16_000, target_num_channels: int = 1, target_format: str = "mp3") -> AudioDecoder:
    # resample audio to desired format
    samples, sample_rate = audioClip.get_all_samples().data, audioClip.metadata.sample_rate
    encoder = AudioEncoder(samples, sample_rate=sample_rate)

    # resample
    resampledData: t.Tensor = encoder.to_tensor(format=target_format, num_channels=target_num_channels, sample_rate=target_sample_rate)

    # return new AudioDecoder
    return AudioDecoder(resampledData)

# plot functions
def waveform(clip: AudioDecoder, title: str = "Waveform Plot"):
    # sample rate & channels
    sr = clip.metadata.sample_rate
    channels = clip.metadata.num_channels

    # get samples
    samples = clip.get_all_samples().data
    nc, ns = samples.shape

    # check
    if nc != channels:
        print(f"Mismatch: reported channels in metadata: {channels} differ from Tensor shape: {samples.shape}")
    
    # time scale (num_samples/sample_rate)
    time_scale = t.arange(0, ns) / sr
    
    fig, axis = plt.subplots(nc,1)
    fig.tight_layout()
    if nc == 1:
        values = samples[0]
        axis.set_xlim([0,time_scale[-1]])
        axis.plot(time_scale, values)
        axis.set_ylabel("Amplitude")
        axis.set_xlabel("Time (s)")
    else:
        for i in range(nc):
            values = samples[i]
            axis[i].set_xlim([0,time_scale[-1]])
            axis[i].plot(time_scale, values)
            axis[i].set_ylabel("Amplitude")
            axis[i].set_xlabel("Time (s)")

    return (fig, axis)

# calculate audio spectrum
def spectrum(clip: AudioDecoder, num_fft_bins: int = FREQUENCY_BINS, title: str = "Power Spectrum"):
    # get samples
    samples = clip.get_all_samples().data
    sample_rate = clip.metadata.sample_rate
    channels = clip.metadata.num_channels

    # spectrum calculator
    s = T.Spectrogram(n_fft=num_fft_bins, power=2)

    # plot
    fig, axis = plt.subplots(channels, 1)
    if channels == 1:
        # calculate spectrum of audio samples (power over frequency)
        spectrogram = s(samples)
    
        # convert signal values to dB 
        samples_dB = T.AmplitudeToDB().forward(spectrogram)

        # plot the spectrogram
        axis.set_ylabel("Freq Bins")
        axis.set_xlabel("Samples")
        axis.imshow(samples_dB.squeeze(), origin="lower", aspect="auto", interpolation="nearest")
    else:
        # calculate spectrogram of each channel
        for i in range(channels):
            spectrogram = s(samples[i])
            # convert signal values to dB 
            samples_dB = T.AmplitudeToDB().forward(spectrogram)

            # plot the spectrogram
            axis[i].set_ylabel("Freq Bins")
            axis[i].set_xlabel("Samples")
            axis[i].imshow(samples_dB.squeeze(), origin="lower", aspect="auto", interpolation="nearest")

    # return the spectrogram
    return (fig, axis)

# convert a tensor into a byte stream
def tensorToBytes(tensor: t.Tensor) -> bytes:
    assert type(tensor) == t.Tensor

    io_obj = BytesIO()

    # save tensor to bytesio
    t.save(tensor, io_obj)

    # read and return bytes
    return io_obj.getvalue()
