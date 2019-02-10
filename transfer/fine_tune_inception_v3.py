'''
a fine-tune demo/sample to classify dog/caa.

It is a modify section
"Fine-tune InceptionV3 on a new set of classes"
from https://keras.io/applications/

The dataset that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data

we need to re-org the data like this:
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the some cat pictures in data/train/cats, and data/validation/cats
- put the some dogs pictures data/train/dogs, data/validation/dogs

here is what I did:

assuming you already downloaded the file: cat_vs_dog_all.zip
mkdir tmp && cd tmp
unzip ../cat_vs_dog_all.zip
unzip train.zip
cd ..
mkdir -p data/train/cats && mkdir -p data/train/dogs
mkdir -p data/validation/cats && mkdir -p data/validation/dogs
mv tmp/train/cat*.* data/train/cats
mv tmp/train/dog*.* data/train/dogs
mv data/train/cats/cat.9*.* data/validation/cats
mv data/train/dogs/dog.9*.* data/validation/dogs

'''
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

# create the base pre-trained model
# load the weight, exclude the last top layer
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 2 classes
predictions = Dense(1, activation='sigmoid')(x)
# if 200 classes, we can use softmax
# predictions = Dense(200, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

model.summary()

# first pass: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
# here we use 2 class, thus binary_crossentropy
model.compile(optimizer='rmsprop', loss='binary_crossentropy',  metrics=['accuracy'])
# if softmax, we will use categorical_crossentropy
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
# load data
#batch_size = 16
#nb_train_samples = 2000
#nb_validation_samples = 800
#epochs = 50
#batch_size = 16

# to speed up for test/demo purpose, I only use very small dataset
# and a small hyper meta parameters
batch_size = 16
nb_train_samples = 200
nb_validation_samples = 40
epochs = 10
batch_size = 16
# dimensions of our images.
img_width, img_height = 150, 150

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')


# load pre-existing weights if exists
#first_pass_weights_file = "first_pass_weights.h5"
#model.load_weights(first_pass_weights_file)

# train the model on the new data for a few epochs
model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples )

model.save_weights(first_pass_weights_file)

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# 2 pass, fine-tune the model

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
# usually use small learning rate for 2-pass
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',  metrics=['accuracy'])
# for softmax
#model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
#2_pass_weights_file = "2_pass_weights.h5"
#model.load_weights(2_pass_weights_file)

model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples )

model.save_weights(2_pass_weights_file)
