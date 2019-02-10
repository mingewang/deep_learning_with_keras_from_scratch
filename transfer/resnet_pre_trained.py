# copy from https://keras.io/applications/#usage-examples-for-image-classification-models
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

# load pre-trained imagenet model 
model = ResNet50(weights='imagenet')

# you can download here wget https://en.wikipedia.org/wiki/File:African_Bush_Elephant.jpg
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
# we need to convert img obj to np array
x = image.img_to_array(img)
# show the image, imshow expect 0-1 range, so we divide 255 here
plt.imshow(x/255.)
plt.show()
# Insert a new axis that will appear at the axis position in the expanded array shape. 
# this is needed as our model need this format
x = np.expand_dims(x, axis=0)
# substract the mean RGB value from each pixel
x = preprocess_input(x)

preds = model.predict(x)
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
decoded = decode_predictions(preds, top=3)
print( decoded )
# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]
