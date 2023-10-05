import librosa
import librosa.display
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

DATASET_PATH = "D:/Isbat/speech_recognition/A4 61 key Piano.wav"

class AudioAnalysis: 
    def __init__(self,filepath):
        signal_array,sr = librosa.load(filepath) #signal gives and array of the audio in a times series. sr stands for SampleRate
        self.signal_array = signal_array
        self.sr = sr
    
    def mel_spectrogram(self):
        mel_spectrogram = librosa.feature.melspectrogram(self.signal_array,
            sr = self.sr,
            n_fft=512,
            hop_length=512,
            n_mels=90)
        self.log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        librosa.display.specshow(self.log_mel_spectrogram,sr=self.sr,
            x_axis = "time",
            y_axis = "mel",)
        plt.title("Mel-Spectrogram",fontname = "Times New Roman",size=16)
        plt.colorbar(format="%+2.f dB")
        plt.show()

    def extract_mfccs(self):
        mfccs = librosa.feature.mfcc(self.signal_array,n_mfcc= 13, sr = self.sr)
        print(mfccs)
        librosa.display.specshow(mfccs,x_axis="time",sr= self.sr)
        plt.title("MFCC", fontname = "Times New Roman", size = 16)
        plt.colorbar(format = "%+2f dB")
        plt.show()
        
if __name__ == "__main__":
    a4_piano = AudioAnalysis(DATASET_PATH)
    a4_piano.mel_spectrogram()
    a4_piano.extract_mfccs()