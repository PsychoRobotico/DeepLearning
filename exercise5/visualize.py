import numpy as np
import matplotlib.pyplot as plt
import dlipr
import keras
from keras.layers import *
from keras import backend as K
import os
from keras.models import *
#from model import *
#import cv2


# Load the Ising dataset
data = dlipr.ising.load_data()

# plot some examples
data.plot_examples(5, fname='examples.png')

# features: images of spin configurations
X_train = data.train_images
X_test = data.test_images

# classes: simulated temperatures
T = data.classes

# labels: class index of simulated temperature
# create binary training labels: T > Tc?
Tc = 2.27
y_train = T[data.train_labels] > Tc
y_test = T[data.test_labels] > Tc

T_index_train=data.train_labels
T_index_test=data.test_labels

Y_train = dlipr.utils.to_onehot(y_train)
Y_test = dlipr.utils.to_onehot(y_test)

B_train=np.reshape(X_train,(22000,32,32,1))
B_test=np.reshape(X_test,(4000,32,32,1))

# ----------------------------------------------------------
# Load your model
# ----------------------------------------------------------
model = keras.models.load_model('model_conv.h5')

print(model.layers)
print model.summary()
conv1 = model.layers[1]
conv2 = model.layers[3]
conv3 = model.layers[5]
conv4 = model.layers[7]

maxPool = model.layers[4]
globalAveragePool = model.layers[8]





print y_train[1]
print Y_train[1]



folder = 'images/'
if not os.path.exists(folder):
    os.makedirs(folder)

for i in range(2):
    a=[]
    a.append(B_test[i])
    a=np.array(a)
    yp=model.predict(a)
    if (round(yp[0][0])==Y_test[i][0]):
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(6, 3.2))
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.98, wspace=0.02)
        dlipr.utils.plot_image(data.test_images[i], ax1)
        ax2.barh(np.arange(2), yp[0], align='center',xerr=Y_test[i])
        ax2.set_yticks(np.arange(2))
        ax2.set_yticklabels(["False","True"])
        ax2.set_xlabel('Propability')
        ax2.set(xlim=(0, 1), xlabel='Probability', yticks=[])
        ax2.text(0.05, 0.1, "False", ha='left', va='center')
        ax2.text(0.05, 1.1, "True", ha='left', va='center')
        ax2.text(0.4,1.3,"T>Tc?",va="center")
        plt.savefig(folder+"image%i.png"%i)








