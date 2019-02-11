# this file
# just save keras model

# import the necessary packages
from keras.applications import ResNet50

model = ResNet50(weights="imagenet") 
model.save('resnet50.h5')



