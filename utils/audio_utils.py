import torch
import torchaudio
from torchaudio.transforms import Resample, Spectrogram

n_fft = 430
hop_length = 160
target_sample_rate = 44100
chunk_size = 14000


def load_audio(file_path):
    samples, rate = torchaudio.load(file_path)
    return prepare_audio(samples, rate)


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
