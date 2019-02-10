# adopted from
# https://github.com/manashmndl/DeadSimpleSpeechRecognizer
import librosa
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# please download google speech commands from
# https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz
# unzip to somewhere 
# also we need to move LICENCE README, _background_noise_ to some other dirs
# in order to use the code here
DATA_PATH = "../data/google_speech_commands/"

# Second dimension of the feature is dim2
feature_dim_2 = 11
# # Feature dimension
feature_dim_1 = 20
channel = 1
epochs = 100
batch_size = 100
verbose = 1
num_classes = 33
weights_file = "asr_hegith.h5"

# Input: Folder Path
# Output: Tuple (Label, Indices of the labels, one-hot encoded labels)
def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

# here we loop each audio dir/class
# convert audio file into mfcc 
# then save into one big array 
def save_data_to_array(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)
    for label in labels:
        # Init mfcc vectors
        mfcc_vectors = []
        wavfiles = [path + label + '/' + wavfile for wavfile in os.listdir(path + '/' + label)]
        # tqdm will make our loops showing a smart progress meter
        for wavfile in tqdm(wavfiles, "Saving vectors of label - '{}'".format(label)):
            mfcc = wav2mfcc(wavfile, max_len=max_len)
            mfcc_vectors.append(mfcc)
        np.save(label + '.npy', mfcc_vectors)

# load data from previous .npy files saved by save_data_to_array()
# sklearn ‘s nice function train_test_split will automatically 
# split the whole dataset.
def get_train_test(split_ratio=0.6, random_state=42):
    # Get available labels
    labels, indices, _ = get_labels(DATA_PATH)
    # Getting first arrays, load from .npy file saved by save_data_to_array()
    X = np.load(labels[0] + '.npy')
    y = np.zeros(X.shape[0])
    # Append all of the dataset into one single array, same goes for y
    for i, label in enumerate(labels[1:]):
        x = np.load(label + '.npy')
        # Stack arrays in sequence vertically (row wise)
        # a = np.array([1, 2, 3]), b= np.array([2, 3, 4] )
        # np.vstack(a,b) => array([[1, 2, 3],
        #                          [2, 3, 4]])
        X = np.vstack((X, x))
        # y will be 0, 1, 2 ... for our classes
        y = np.append(y, np.full(x.shape[0], fill_value= (i + 1)))
    assert X.shape[0] == len(y)
    return train_test_split(X, y, test_size= (1 - split_ratio), random_state=random_state, shuffle=True)

# prepare data ( save raw audio file into .npy files) 
# just need to do once
def pre_process_data():
  # Save data to array file first
  save_data_to_array(max_len=feature_dim_2)

# define our model
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(2, 2), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model

def train(model, pre_load):
  if pre_load:
    model.load_weights(weights_file)
  # load train set and test set
  X_train, X_test, y_train, y_test = get_train_test()
  # Reshaping to perform 2D convolution
  X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
  X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)
  # one-hot-encoded for softmax
  y_train_hot = to_categorical(y_train)
  y_test_hot = to_categorical(y_test)
  # train
  model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
  # save the result
  model.save_weights(weights_file)

# predicts a audio file at filepath
def predict(filepath, model):
    # load
    model.load_weights(weights_file)
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

# just need to pre-process once
pre_process_data()
model = get_model()
model.summary()
train(model, False)
#train(model, True)

# testing real audio file
# you can record audio as wav file with this format:
# RIFF (little-endian) data, WAVE audio, Microsoft PCM, 16 bit, mono 16000 Hz
# you can use sox utility to convert your audio to this format
# predict_text = predict('clip10.wav', model=model)
# print( predict_text )
