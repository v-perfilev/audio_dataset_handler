from torch.utils.data import Dataset

from torch.utils.data import Dataset

from utils.audio_utils import load_audio, sample_to_spectrogram


class SpeechPhonemesDataset(Dataset):

    def __init__(self, file_phoneme_tuples, limit=None):
        if limit is not None:
            file_phoneme_tuples = file_phoneme_tuples[:limit]
        self.data = self.__create_data(file_phoneme_tuples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def __create_data(file_phoneme_tuples):
        data = []

        for (file, phoneme_count) in file_phoneme_tuples:
            sample, _ = load_audio(file)
            spectrogram = sample_to_spectrogram(sample)
            data.append((spectrogram, phoneme_count))

        return data
