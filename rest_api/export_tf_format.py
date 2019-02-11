# this file will load keras model, 
# then export a SavedModel format for tensorflow serving

import tensorflow as tf

# The export path contains the name and the version of the model
tf.keras.backend.set_learning_phase(False)
model = tf.keras.models.load_model('./resnet50.h5')
export_path = './my_image_classifier/2'

# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors
# And stored with the default serving key
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_image': model.input},
        outputs={t.name: t for t in model.outputs})

