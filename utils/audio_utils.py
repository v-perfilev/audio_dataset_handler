import os

import noisereduce as nr
import torch
import torchaudio
from pydub.generators import WhiteNoise
from torchaudio.transforms import Resample, Spectrogram

n_fft = 430
hop_length = 160
target_sample_rate = 44100
chunk_size = 14000
noise_duration = 3000

desired_dBFS = -20
desired_rms = 10 ** (desired_dBFS / 20.0)


def load_audio(file_path):
    samples, rate = torchaudio.load(file_path)
    return prepare_audio(samples, rate)


def generate_noise(filename, path='tmp'):
    noise = WhiteNoise().to_audio_segment(duration=3000).apply_gain(-20)
    os.makedirs(path, exist_ok=True)
    noise.export(f"{path}/{filename}", format="wav")


def prepare_audio(samples, rate):
    if samples.dim() > 1 and samples.shape[0] == 2:
        samples = samples.mean(dim=0, keepdim=True)
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)

    if rate != target_sample_rate:
        resample_transform = Resample(orig_freq=rate, new_freq=target_sample_rate)
        samples = resample_transform(samples)
        rate = target_sample_rate

    return samples, rate


def mix_audio_samples(main_waveform, background_waveform, background_volume):
    background_waveform *= background_volume

    if main_waveform.shape[1] > background_waveform.shape[1]:
        repeat_times = main_waveform.shape[1] // background_waveform.shape[1] + 1
        background_waveform = background_waveform.repeat(1, repeat_times)
    background_waveform = background_waveform[:, :main_waveform.shape[1]]

    mixed_waveform = main_waveform + background_waveform

    return mixed_waveform


def denoise(sample, rate):
    sample = nr.reduce_noise(sample, rate)
    return torch.from_numpy(sample)


def normalize(sample):
    current_rms = torch.sqrt(torch.mean(sample ** 2))
    if current_rms > 0:
        scaling_factor = desired_rms / current_rms
        sample = sample * scaling_factor
    return sample


def divide_audio(audio):
    chunks = audio.squeeze(0).unfold(0, chunk_size, chunk_size).contiguous()
    processed_chunks = []
    for chunk in chunks:
        if chunk.size(0) < chunk_size:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_size - chunk.size(0)))
        processed_chunks.append(chunk.unsqueeze(0))
    return processed_chunks


def compile_audio(chunks):
    return torch.cat(chunks, dim=0)


def sample_to_spectrogram(sample):
    spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length)
    return spectrogram(sample)


def save_audio(audio_data, filename, path="target"):
    os.makedirs(path, exist_ok=True)
    torchaudio.save(path + "/" + filename, audio_data, target_sample_rate)
