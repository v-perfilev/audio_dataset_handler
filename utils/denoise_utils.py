import noisereduce as nr
import torch


def denoise_waveform(waveform, rate):
    """Denoise waveform with noisereduce library."""
    waveform = nr.reduce_noise(waveform, rate)
    return torch.from_numpy(waveform)
