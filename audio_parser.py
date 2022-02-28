import torch
import librosa
import torchaudio


class MELspParser(object):
    def __init__(self, sr, n_fft, win_length, hop_length, n_mels, top_db=150):
        super(MELspParser, self).__init__()
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = hop_length
        self.sample_rate = sr
        self.n_mels = n_mels
        self.top_db = top_db

        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                            n_fft=self.n_fft,
                                                            win_length=self.win_length,
                                                            hop_length=self.hop_length,
                                                            n_mels=self.n_mels)

    def __call__(self, audio):
        # melspectrogram features, as FSD50k paper
        x = torch.from_numpy(audio).unsqueeze(0)
        specgram = self.melspec(x)[0].numpy()
        specgram = librosa.power_to_db(specgram)
        specgram = specgram.astype('float32')
        specgram += self.top_db / 2
        mean = specgram.mean()
        specgram -= mean   # per sample zero centering
        return specgram
