import gc

import torch
from torch.utils.data import Dataset

from utils.audio_utils import load_waveform, waveform_to_spectrogram, divide_spectrogram, compile_spectrogram

# Load speech denoiser model
model_path = "../_models/speech_denoiser_model_scripted.pth"
model = torch.jit.load(model_path)
model.eval()


class ValentiniSpeechPhonemesDataset(Dataset):
    """Dataset for loading speech audio files and their corresponding phoneme counts."""

    def __init__(self, file_phoneme_tuples, limit=None):
        """Initialize the dataset with a list of tuples containing file paths and phoneme counts."""
        # Apply limit if specified to restrict the dataset size
        if limit is not None:
            file_phoneme_tuples = file_phoneme_tuples[:limit]
        # Process the tuples to create a list of data points
        self.data = self.__create_data(file_phoneme_tuples)

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a data pair consisting of a spectrogram and its corresponding phoneme count by index."""
        return self.data[idx]

    @staticmethod
    def __create_data(file_phoneme_tuples):
        """Process each file and phoneme count tuple to create spectrograms."""
        data = []

        # Process each file-phoneme tuple to create spectrogram and store with phoneme count
        for idx, (file, phoneme_count) in enumerate(file_phoneme_tuples):
            waveform, _ = load_waveform(file)
            spectrogram = waveform_to_spectrogram(waveform)
            spectrograms, spectrogram_length = divide_spectrogram(spectrogram)
            with torch.no_grad():
                spectrograms = model(spectrograms)
            spectrogram = compile_spectrogram(spectrograms, spectrogram_length)
            data.append((spectrogram, phoneme_count))

            del waveform, spectrogram, spectrograms
            gc.collect()

            # Log progress every 100 files
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1} files from {len(file_phoneme_tuples)}")

        return data
