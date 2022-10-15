import whisper
import jiwer

from whisper.normalizers import EnglishTextNormalizer
import torchaudio

#https://pytorch.org/audio/0.13.0/datasets.html

class Whisper():
    '''
    Helper class for Open AI whisper module.

    '''

    def __init__(self):
        self.model = whisper.load_model("base.en")
        self.options = whisper.DecodingOptions()
    
    @staticmethod
    def format_audio(audio_path):
        '''
        Takes the raw audio input tensor and transforms it to format required by whisper.

        Parameters
        ----------
        audio_path: str

        Returns
        ---------

        formated_audio: tensor
        

        '''
        audio, _ = torchaudio.load(audio_path)
        return whisper.pad_or_trim(audio.flatten())
    
    @staticmethod
    def wer(hypothesis, ground_truth):
        '''
        The word error rate of the whisper transcript compared to the true transcript.

        Parameters
        ----------

        hypothesis: str
            The transcript from whisper

        ground_truth: str
            The true transcript


        Returns
        ----------
        word error rate: float
            
        '''
        normalizer = EnglishTextNormalizer()
        return jiwer.wer(normalizer(hypothesis), normalizer(ground_truth))

    def transcribe(self, audio_path):
        '''
        Transcription of audio

        Parameters
        ----------
        audio_path: str
            path to audio file (.flac)
            
        Returns
        ---------
        transcript: dict
        
        '''
        # add check audio type/conversion here 

        _audio = self.format_audio(audio_path)
        return self.model.transcribe(_audio)
    
    def log_mel_spectrogram(self, audio):
        '''
        Creates the log mel spectrogram from given audio input

        Parameters
        ----------
        audio: tensor

        Returns
        ---------
        mel_spec: tensor

        '''
        _audio = self.format_audio(audio)
        mel = whisper.log_mel_spectrogram(_audio).to(self.model.device) # create the log-mel spectrogram
        return mel
    
    #include raw decode from mel spec - does not work on my CPU.
    #whisper.decode(model, mel, options))
    
