import gc
import random

import torch
from torch.utils.data import Dataset

from utils.audio_utils import load_waveform, waveform_to_spectrogram, divide_spectrogram, compile_spectrogram

# Load speech denoiser model
model_path = "../_models/speech_denoiser_model_scripted.pth"
model = torch.jit.load(model_path)
model.eval()


class ValentiniSpeechSyllablesDataset(Dataset):
    """Dataset for loading speech audio files and their corresponding syllable counts."""

    def __init__(self, file_syllable_tuples, limit=None):
        """Initialize the dataset with a list of tuples containing file paths and syllable counts."""
        # Apply limit if specified to restrict the dataset size
        if limit is not None:
            random.shuffle(file_syllable_tuples)
            file_syllable_tuples = file_syllable_tuples[:limit]
        # Process the tuples to create a list of data points
        self.data = self.__create_data(file_syllable_tuples)

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a data pair consisting of a spectrogram and its corresponding syllable count by index."""
        return self.data[idx]

    @staticmethod
    def __create_data(file_syllable_tuples):
        """Process each file and syllable count tuple to create spectrograms."""
        data = []

        # Process each file-syllable tuple to create spectrogram and store with syllable count
        for idx, (file, syllable_count) in enumerate(file_syllable_tuples):
            waveform, _ = load_waveform(file)
            spectrogram = waveform_to_spectrogram(waveform)
            spectrograms, spectrogram_length = divide_spectrogram(spectrogram)
            with torch.no_grad():
                spectrograms = model(spectrograms)
            spectrogram = compile_spectrogram(spectrograms, spectrogram_length)
            data.append((spectrogram, syllable_count))

            del waveform, spectrogram, spectrograms

            # Log progress every 100 files
            if (idx + 1) % 100 == 0:
                gc.collect()
                print(f"Processed {idx + 1} files from {len(file_syllable_tuples)}")

        return data
