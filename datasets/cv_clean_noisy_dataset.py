import random

from torch.utils.data import Dataset

from utils.audio_utils import load_waveform, mix_waveforms, divide_waveform, waveform_to_spectrogram
from utils.denoise_utils import denoise_waveform


class CvCleanNoisyDataset(Dataset):
    """A dataset class for creating spectrograms from clean and noisy audio files for machine learning models."""

    def __init__(self, clean_files, sound_files, limit=None):
        """Initialize the dataset with file lists and an optional limit on the number of files to process."""
        # Limit the number of clean files if a limit is specified
        if limit is not None:
            clean_files = clean_files[:limit]
        # Preprocess the files to create spectrograms
        self.spectrograms = self.__create_spectrograms(clean_files, sound_files)

    def __len__(self):
        """Return the number of spectrogram pairs in the dataset."""
        return len(self.spectrograms)

    def __getitem__(self, idx):
        """Retrieve a spectrogram pair by index."""
        return self.spectrograms[idx]

    @staticmethod
    def __create_spectrograms(clean_files, sound_files):
        """Generate pairs of clean and noisy spectrograms from audio files."""
        clean_spectrograms = []
        noisy_spectrograms = []

        # Process each file to create spectrograms
        for idx, clean_file in enumerate(clean_files):
            sound_file = random.choice(sound_files)  # Select a random noise file
            clean_sample, clean_rate = load_waveform(clean_file)
            sound_sample, _ = load_waveform(sound_file)

            # Determine volume of the noise to be added
            background_volume = CvCleanNoisyDataset.__choose_noise_volume()
            noisy_sample = mix_waveforms(clean_sample, sound_sample, background_volume)

            # Denoise clean sample
            clean_sample = denoise_waveform(clean_sample, clean_rate)

            # Divide the audio into chunks and convert to spectrograms
            clean_sample_chunks = divide_waveform(clean_sample)
            for clean_sample_chunk in clean_sample_chunks:
                clean_spectrogram = waveform_to_spectrogram(clean_sample_chunk)
                clean_spectrograms.append(clean_spectrogram)

            noisy_sample_chunks = divide_waveform(noisy_sample)
            for noisy_sample_chunk in noisy_sample_chunks:
                noisy_spectrogram = waveform_to_spectrogram(noisy_sample_chunk)
                noisy_spectrograms.append(noisy_spectrogram)

            # Log progress every 100 files
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} files from {len(clean_files)}, total chunks: {len(clean_spectrograms)}")

        return list(zip(noisy_spectrograms, clean_spectrograms))

    @staticmethod
    def __choose_noise_volume(min_volume=0.2):
        """Randomly choose a volume for the noise to be added to the clean signal."""
        # Generate a list of possible volume levels and select one
        return random.choice([i / 10 + min_volume for i in range(0, 5)])
