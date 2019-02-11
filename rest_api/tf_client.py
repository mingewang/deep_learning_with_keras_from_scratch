# a sample client to TF Serving
import argparse
import json

from keras.applications import ResNet50
import numpy as np
import requests
from keras.preprocessing import image
from keras.applications import imagenet_utils

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) 

# this line is added because of a bug in tf_serving(1.10.0-dev)
#img = img.astype('float16')

payload = {
    "instances": [{'input_image': img.tolist()}]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:8501/v1/models/resnet:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))
#print( pred )

# Decoding the response
result = imagenet_utils.decode_predictions( np.array(pred["predictions"]) )
print( result )
