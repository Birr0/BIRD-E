import os

try:
    import tensorflow 
except ImportError:
    pass

import torch
import torchaudio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        #audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        #mel = whisper.log_mel_spectrogram(audio)
        return  self.dataset[item] #(mel, text)

dataset = LibriSpeech("test-clean")

loader = torch.utils.data.DataLoader(dataset, batch_size=16)
