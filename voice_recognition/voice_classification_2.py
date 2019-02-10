#%load_ext autoreload
#%autoreload 2

from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization 
from keras.utils import to_categorical

# Second dimension of the feature is dim2
feature_dim_2 = 11

# # Feature dimension
feature_dim_1 = 20
channel = 1
epochs = 100
batch_size = 100
verbose = 1
num_classes = 33


def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', strides=1, activation='relu', input_shape=(feature_dim_1, feature_dim_2, channel)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', strides=1, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))
    model.add(BatchNormalization())

#    model.add(Conv2D(128, kernel_size=(2, 2), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(256, kernel_size=(2, 2), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))

#    model.add(Dropout(0.25))
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

# Predicts one sample
def predict(filepath, model):
    model.load_weights("asr_hegith_2.h5")
    sample = wav2mfcc(filepath)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))
    ]

def pre_proc_data():
  # Save data to array file first
  # prepare data
  save_data_to_array(max_len=feature_dim_2)

def train(model, pre_load):
  if pre_load:
    model.load_weights("asr_hegith_2.h5")
  # # Loading train set and test set
  X_train, X_test, y_train, y_test = get_train_test()
  # Reshaping to perform 2D convolution
  X_train = X_train.reshape(X_train.shape[0], feature_dim_1, feature_dim_2, channel)
  X_test = X_test.reshape(X_test.shape[0], feature_dim_1, feature_dim_2, channel)

  y_train_hot = to_categorical(y_train)
  y_test_hot = to_categorical(y_test)
  model.fit(X_train, y_train_hot, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_test, y_test_hot))
  model.save_weights("asr_hegith_2.h5")


#from keras.models import load_model
model = get_model()
model.summary()
train(model, True)


#model = load_model('asr_hegith.h5') 
#print(predict('/home/mwang/Recordings/clip8.wav', model=model))
print(predict('/home/mwang/Recordings/clip10.wav', model=model))
#print(predict('../data/google_speech_commands/one/00176480_nohash_0.wav', model=model))
