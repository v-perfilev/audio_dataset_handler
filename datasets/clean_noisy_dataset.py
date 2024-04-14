import random

from torch.utils.data import Dataset

from utils.audio_utils import load_audio, mix_audio_samples, divide_audio, sample_to_spectrogram, generate_noise, \
    denoise, save_audio, normalize


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

        generate_noise("noise.wav")
        noise_sound, _ = load_audio("tmp/noise.wav")

        for idx, clean_file in enumerate(clean_files):

            sound_file = random.choice(sound_files)
            clean_sample, clean_rate = load_audio(clean_file)
            clean_sample = denoise(normalize(clean_sample), clean_rate)
            sound_sample, _ = load_audio(sound_file)

            background_volume = CleanNoisyDataset.__choose_noise_volume()
            noisy_sample = mix_audio_samples(clean_sample, sound_sample, background_volume)

            noise_volume = CleanNoisyDataset.__choose_noise_volume()
            noisy_sample = mix_audio_samples(clean_sample, noisy_sample, noise_volume)

            save_audio(noisy_sample, "noisy.wav")
            save_audio(clean_sample, "clean.wav")

            clean_sample_chunks = divide_audio(clean_sample)
            for clean_sample_chunk in clean_sample_chunks:
                clean_spectrogram = sample_to_spectrogram(clean_sample_chunk)
                clean_spectrograms.append(clean_spectrogram)

            noisy_sample_chunks = divide_audio(noisy_sample)
            for noisy_sample_chunk in noisy_sample_chunks:
                noisy_spectrogram = sample_to_spectrogram(noisy_sample_chunk)
                noisy_spectrograms.append(noisy_spectrogram)

            if idx % 100 == 0:
                print(f"Processed {idx} files from {len(clean_files)}")

        return list(zip(clean_spectrograms, noisy_spectrograms))

    @staticmethod
    def __choose_noise_volume(min_volume=0.2):
        return random.choice([i / 10 + min_volume for i in range(0, 5)])
