import random

from torch.utils.data import Dataset

from utils.audio_utils import load_audio, mix_audio_samples, divide_audio, sample_to_spectrogram


class CleanNoisyDataset(Dataset):

    def __init__(self, clean_files, sound_files, limit=None):
        if limit is not None:
            clean_files = clean_files[:limit]
        self.spectrograms = self.__create_spectrograms(clean_files, sound_files)

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx]

    @staticmethod
    def __create_spectrograms(clean_files, sound_files):
        clean_spectrograms = []
        noisy_spectrograms = []

        for clean_file in clean_files:
            background_volume = random.choice([i / 10 + 0.2 for i in range(1, 5)])
            sound_file = random.choice(sound_files)
            clean_sample, _ = load_audio(clean_file)
            sound_sample, _ = load_audio(sound_file)
            noisy_sample = mix_audio_samples(clean_sample, sound_sample, background_volume)

            clean_sample_chunks = divide_audio(clean_sample)
            for clean_sample_chunk in clean_sample_chunks:
                clean_spectrogram = sample_to_spectrogram(clean_sample_chunk)
                clean_spectrograms.append(clean_spectrogram)

            noisy_sample_chunks = divide_audio(noisy_sample)
            for noisy_sample_chunk in noisy_sample_chunks:
                noisy_spectrogram = sample_to_spectrogram(noisy_sample_chunk)
                noisy_spectrograms.append(noisy_spectrogram)

        return list(zip(clean_spectrograms, noisy_spectrograms))
