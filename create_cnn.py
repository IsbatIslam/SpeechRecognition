#Importing Libraries
import librosa
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

#Constants
DATA_PATH = "/content/drive/MyDrive/audio_data.json" 
MODEL_PATH = "model.h5"
LEARNING_RATE = 0.0001
EPOCHS=40 #EPOCH refers to how many times the dataset is passed through the model, back and forth. Here we want 40 times.
BATCH_SIZE = 32 #How many training samples we want per batch
NUM_KEYWORDS = 5 #One, two, three, four and five

#Load Dataset
def load_dataset(data_path):
    with open(data_path,"r") as fp:
        data = json.load(fp)

    #extract MFCC(X) and Labels (Y)
    X = np.array(data["MFCCs"])
    Y = np.array(data["labels"])
    return X,Y

#Prepare training, validation and testing dataset
def get_data_splits(data_path, test_size=0.1, test_validation = 0.1):

    X,Y = load_dataset(data_path)

    #create train/validation/test splits:
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=test_size)
    X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train, test_size = test_validation)

    #convert inputs from 2d to 3d arrays
    X_train = X_train[...,np.newaxis] #[...,np.newaxis] is used to create another dimension as the CNN model takes in 3 dimension.
    X_validation = X_validation[...,np.newaxis]
    X_test = X_test[...,np.newaxis]

    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

#Building the model
def build_model(input_shape,learning_rate,error = "sparse_categorical_crossentropy"):
    
    #build the network
    model = keras.Sequential()
    
    #conv layer 1
    model.add(keras.layers.Conv2D(64,(3,3),activation= "relu",input_shape = input_shape,kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3),strides = (2,2), padding = "same"))

    #conv layer 2
    model.add(keras.layers.Conv2D(32,(3,3),activation= "relu",
    kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3),strides = (2,2), padding = "same"))

    #conv layer 3
    model.add(keras.layers.Conv2D(32,(2,2),activation= "relu",
    kernel_regularizer = keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2),strides = (2,2), padding = "same"))

    #flatten the output and feed it into a dense layer. Flatten means to take the matrix, and put it all values into 1 column.
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = "relu"))
    model.add(keras.layers.Dropout(0.3)) #Dropout reduces overfitting.
    model.add(keras.layers.Dense(NUM_KEYWORDS,activation = "softmax"))

    #compile the model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,loss=error,metrics = ["accuracy"])

    #print model overview
    model.summary()

    return model

#Running the whole program
def main():

    #load train/validation/testdata splits
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = get_data_splits(DATA_PATH)
    
    #build CNN model
    input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]) # (#segments (samples/hop_length),#coefficients(13),#channels, mono (1))
    model = build_model(input_shape,LEARNING_RATE)

    #train the model
    model.fit(X_train,Y_train,epochs = EPOCHS, batch_size = BATCH_SIZE,
    validation_data = (X_validation,Y_validation)) 

    #evaluate the model
    test_error,test_accuracy = model.evaluate(X_test,Y_test)
    print(f"\n\nTest error: {test_error},Test accuracy:{test_accuracy}")

    #save the model
    model.save(MODEL_PATH)

if __name__ == "__main__":
    main()