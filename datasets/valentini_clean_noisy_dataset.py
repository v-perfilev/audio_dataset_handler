from torch.utils.data import Dataset

from utils.audio_utils import load_audio, divide_audio, sample_to_spectrogram


class ValentiniCleanNoisyDataset(Dataset):
    """A dataset class for creating spectrograms from clean and noisy audio files for machine learning models."""

    def __init__(self, clean_files, noisy_files, limit=None):
        """Initialize the dataset with file lists and an optional limit on the number of files to process."""
        # Limit the number of clean files if a limit is specified
        if limit is not None:
            clean_files = clean_files[:limit]
        # Preprocess the files to create spectrograms
        self.spectrograms = self.__create_spectrograms(clean_files, noisy_files)

    def __len__(self):
        """Return the number of spectrogram pairs in the dataset."""
        return len(self.spectrograms)

    def __getitem__(self, idx):
        """Retrieve a spectrogram pair by index."""
        return self.spectrograms[idx]

    @staticmethod
    def __create_spectrograms(clean_files, noisy_files):
        """Generate pairs of clean and noisy spectrograms from audio files."""
        clean_spectrograms = []
        noisy_spectrograms = []

        # Process each file to create spectrograms
        for idx, _ in enumerate(clean_files):
            clean_file = clean_files[idx]
            noisy_file = noisy_files[idx]

            clean_sample, clean_rate = load_audio(clean_file)
            noisy_sample, noisy_rate = load_audio(noisy_file)

            # Divide the audio into chunks and convert to spectrograms
            clean_sample_chunks = divide_audio(clean_sample)
            for clean_sample_chunk in clean_sample_chunks:
                clean_spectrogram = sample_to_spectrogram(clean_sample_chunk)
                clean_spectrograms.append(clean_spectrogram)

            noisy_sample_chunks = divide_audio(noisy_sample)
            for noisy_sample_chunk in noisy_sample_chunks:
                noisy_spectrogram = sample_to_spectrogram(noisy_sample_chunk)
                noisy_spectrograms.append(noisy_spectrogram)

            # Log progress every 100 files
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} files from {len(clean_files)}")

        return list(zip(clean_spectrograms, noisy_spectrograms))
