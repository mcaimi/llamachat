#!/usr/bin/env python

try:
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder
    import matplotlib.pyplot as plt
    import torchaudio as ta
    import torchaudio.transforms as T
    from torch import Tensor, arange
    import numpy as np
except ImportError as e:
    raise(e)

# constants and configurations
TARGET_SAMPLING_RATE = 16_000
TARGET_NUM_CHANNEL = 1
TARGET_FORMAT="mp3"
FREQUENCY_BINS = 512

# resample a clip
def resample(audioClip: AudioDecoder, target_sample_rate: int = TARGET_SAMPLING_RATE,
            target_num_channels: int = TARGET_NUM_CHANNEL, target_format: str = TARGET_FORMAT) -> AudioDecoder:
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
    time_scale = arange(0, ns) / sr
    
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
        mono_data = samples
        spectrogram = s(mono_data)
    
        # convert signal values to dB 
        samples_dB = T.AmplitudeToDB().forward(spectrogram)

        # plot the spectrogram
        axis.set_ylabel("Freq Bins")
        axis.set_xlabel("Time")
        axis.imshow(samples_dB.squeeze(), origin="lower", aspect="auto", interpolation="nearest")
    else:
        # calculate spectrogram of each channel
        for i in range(channels):
            spectrogram = s(samples[i])
            # convert signal values to dB 
            samples_dB = T.AmplitudeToDB().forward(spectrogram)

            # plot the spectrogram
            axis[i].set_ylabel("Freq Bins")
            axis[i].set_xlabel("Time")
            axis[i].imshow(samples_dB.squeeze(), origin="lower", aspect="auto", interpolation="nearest")

    # return the spectrogram
    return (fig, axis)