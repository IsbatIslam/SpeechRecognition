import librosa
import os
import json

DATASET_PATH = "D:/Isbat/speech_recognition/audio_data_wav" 
JSON_PATH = "audio_data.json" #the JSON file that we will create
SAMPLES = 22050 #The Nyquist frequency of 44.1KHz. This is set to 22 050 samples/second

def prepare_dataset(dataset_path, json_path,n_mfcc=13,hop_length = 512,n_fft = 2048):
    
    #Dictionnary to store info about what the digits are, and 
    data = {
        "digits": [],
        "labels":[],
        "MFCCs":[],
        }
        
    #loop through all the the folders and files to store data
    for i, (dirpath,dirnames,filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:

            #save the digit names
            dirpath_components = dirpath.split("\\") #cuts the head from the tail: [...]/audio_data_wav/1one --> [/audio_data_wav,1one]
            semantic_label = dirpath_components[-1] 
            data['digits'].append(semantic_label) #Store the digit name.

            #extract mfcc for each audio files
            for file in filenames:
                file_path = os.path.join(dirpath,file)
                signal,sr = librosa.load(file_path)

                #uniformalize all signals to 1 second
                signal = signal[:SAMPLES]

                #extract mfcc. The parameters are all set in prepare_dataset()
                mfcc = librosa.feature.mfcc(signal,n_mfcc=n_mfcc,
                n_fft = n_fft,hop_length=hop_length)

                #store data
                data["labels"].append(i-1) #because of this code, all the labels will be stored as one less than their actual values,
                #but that's to keep it from making an error later as you will have 5 labels, but [0,5] gives 6 indexes, which causes an error.
                data["MFCCs"].append(mfcc.T.tolist())

    #Save everything inside of a JSON file               
    with open(json_path,"w") as fp:
        json.dump(data,fp,indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH,JSON_PATH)