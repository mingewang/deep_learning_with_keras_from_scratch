How to run TF Serving

docker pull tensorflow/serving

# export keras model
python export_keras_model.py

# export SavedModel format
python export_tf_format.py 

# start tf serving server
docker run -p 8501:8501 --name tfserving_resnet \
-v "$(pwd)/my_image_classifier:/models/resnet" \
-e MODEL_NAME=resnet -t tensorflow/serving &
