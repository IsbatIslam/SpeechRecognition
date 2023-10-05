#Importing Libraries
import librosa
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Constants
MODEL_PATH = "/content/drive/MyDrive/model.h5"
#DATATEST_PATH = "D:/Isbat/speech_recognition/audio_test_voice"
SAMPLES = 22050

class _Keyword_Spotting_Service:

    model = None

    mapping = [
    "one",
    "two",
    "three",
    "four",
    "five"]

    _instance = None
    
    #gets mfcc of the test audio
    def get_mfcc(self,file_path,n_mfcc=13,n_fft=2048,hop_length = 512):
        #load audio file
        signal, sr = librosa.load(file_path)
        signal = signal[:SAMPLES] #Uniformalize all signal
        #extract MFCCs
        MFCCs = librosa.feature.mfcc(signal,n_mfcc=n_mfcc,n_fft=n_fft,hop_length=hop_length)
        return MFCCs.T

    #predicts the label of the audio
    def predict(self,file_path):
        #extract the MFCC's
        MFCCs = self.get_mfcc(file_path)
        #convert 2d mfcc's into 4d arrays
        MFCCs = MFCCs[np.newaxis,...,np.newaxis]
        #make predicition
        prediction = self.model.predict(MFCCs)
        predicted_index = np.argmax(prediction)
        predicted_digit = self.mapping[predicted_index]
        return predicted_digit

def Keyword_Spotting_Service():
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH) #our CNN model
    return _Keyword_Spotting_Service._instance


# THIS PART IS FOR ANALYSIS
def accuracy_gauge(data_path):

    testset ={ "label":['one','two','three','four','five'],
    "predicted_digit" :[]}

    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(data_path)):
        if dirpath is not data_path:
            for f in filenames:
                file_path = os.path.join(dirpath,f)
                testset["predicted_digit"].append(kss.predict(file_path))
    print(testset)

if __name__ == "__main__":
    kss = Keyword_Spotting_Service()
    digit1 = kss.predict("/content/drive/MyDrive/five.005.wav")
    #digit2 = kss.predict("/content/drive/MyDrive/one.001.wav")
    print(f'You said "{digit1}"" and "{digit2}".')
