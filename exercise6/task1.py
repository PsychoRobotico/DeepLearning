import numpy as np
import keras
import dlipr
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import matplotlib.pyplot as plt
import os
import numpy
# ------------------------------------------------------------
# Apply a trained ImageNet classification network to classify new images.
# See https://keras.io/applications/ for further instructions.
# ------------------------------------------------------------

# Note: Keras downloads the pretrained network weights to ~/.keras/models.
# To save space in your home folder you can use the /net/scratch/deeplearning/keras-models folder.
# Simply open the terminal and copy/paste:
# ln -s /net/scratch/deeplearning/keras-models ~/.keras/models
# If you get an error "cannot overwrite directory", remove the existing .keras/models folder first.
# Alternatively, you can set up the model with "weights=None" and then use model.load_weights('/net/scratch/deeplearning/keras-models/...')

# Example: Inception-v3
#from keras.applications.inception_v3 import InceptionV3
#model = InceptionV3(weights='imagenet')
from keras.applications.resnet50 import ResNet50
model = ResNet50(weights='imagenet')

folder = 'task1/'
if not os.path.exists(folder):
    os.makedirs(folder)

img_path = folder+'formula12.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print'Predicted: '
for i in decode_predictions(preds, top=10)[0]:
    print i