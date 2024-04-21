import os

import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram

import config


def load_audio(file_path):
    """Load an audio file into a tensor using torchaudio."""
    samples, rate = torchaudio.load(file_path)
    return prepare_audio(samples, rate)


def prepare_audio(samples, rate):
    """Prepare audio for processing: handle mono/stereo and resample if necessary."""
    # Convert stereo to mono by averaging the two channels
    if samples.dim() > 1 and samples.shape[0] == 2:
        samples = samples.mean(dim=0, keepdim=True)
    # Ensure the sample has a batch dimension
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    # Resample to the desired sample rate if necessary
    if rate != config.SAMPLE_RATE:
        resample_transform = Resample(orig_freq=rate, new_freq=config.SAMPLE_RATE)
        samples = resample_transform(samples)
        rate = config.SAMPLE_RATE
    return samples, rate


def mix_audio_samples(main_waveform, background_waveform, background_volume):
    """Mix two audio samples with a specified volume for the background."""
    # Apply volume adjustment to the background waveform
    background_waveform *= background_volume
    # Repeat background to match the length of the main waveform
    if main_waveform.shape[1] > background_waveform.shape[1]:
        repeat_times = main_waveform.shape[1] // background_waveform.shape[1] + 1
        background_waveform = background_waveform.repeat(1, repeat_times)
    background_waveform = background_waveform[:, :main_waveform.shape[1]]
    # Mix both waveforms
    mixed_waveform = main_waveform + background_waveform
    return mixed_waveform


def normalize(sample):
    """Normalize audio sample to have a desired RMS."""
    current_rms = torch.sqrt(torch.mean(sample ** 2))
    if current_rms > 0:
        scaling_factor = config.DESIRED_RMS / current_rms
        sample = sample * scaling_factor
    return sample


def divide_audio(audio):
    """Divide audio into chunks of a predetermined size, padding if necessary."""
    chunks = audio.squeeze(0).unfold(0, config.CHUNK_SIZE, config.CHUNK_SIZE).contiguous()
    processed_chunks = []
    for chunk in chunks:
        # Pad the chunk if it's less than the full chunk size
        if chunk.size(0) < config.CHUNK_SIZE:
            chunk = torch.nn.functional.pad(chunk, (0, config.CHUNK_SIZE - chunk.size(0)))
        processed_chunks.append(chunk.unsqueeze(0))
    return processed_chunks


def compile_audio(chunks):
    """Compile chunks of audio back into a single audio tensor."""
    return torch.cat(chunks, dim=0)


def sample_to_spectrogram(sample):
    """Convert an audio sample to a spectrogram with given FFT and hop length."""
    spectrogram = Spectrogram(n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return spectrogram(sample)


def save_audio(audio_data, filename, path="target"):
    """Save audio data to a file in a specified directory."""
    os.makedirs(path, exist_ok=True)
    torchaudio.save(path + "/" + filename, audio_data, config.SAMPLE_RATE)
