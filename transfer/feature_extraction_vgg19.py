# copy from https://keras.io/applications
# with minor modification and more document

from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np

base_model = VGG19(weights='imagenet')

base_model.summary() 

# we have to watch carefully about base_model.summary() 
# then we can choose which feature/layer we want
#
# now we can setup a new model, thus we could get our desired feature
# the block4_pool comes one layer output from base_model.summary()
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# if we have many images, we can get features for all those images
block4_pool_features = model.predict(x)

print( block4_pool_features )

# we can do something in the features space
