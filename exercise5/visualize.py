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
import scipy.ndimage

import tensorflow as tf

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
"""
conv1 = model.layers[1]
conv2 = model.layers[3]
conv3 = model.layers[5]
conv4 = model.layers[7]
"""
conv1 = model.layers[0]
act1  = model.layers[1]
conv2 = model.layers[4]
act2 = model.layers[5]
avepool=model.layers[6]

"""
maxPool = model.layers[4]
globalAveragePool = model.layers[8]
"""
print "c1", conv1.get_weights()[0].shape
print "c2", conv2.get_weights()[0].shape
#print "c3", conv3.get_weights()[0].shape
#print "c4", conv4.get_weights()[0].shape



#print y_train[1]
#print Y_train[1]



folder = 'images/'
if not os.path.exists(folder):
    os.makedirs(folder)

for i in range(350):
    a=[]
    a.append(B_test[i])
    a=np.array(a)
    yp=model.predict(a)
    if not(round(yp[0][0])==Y_test[i][0]):
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


"""
def visualize_activation(A, name='conv'):
    nx, ny, nf = A.shape
    n = np.ceil(nf**.5).astype(int)
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
    for i in range(n**2):
        ax = axes.flat[i]
        if i < nf:
            ax.imshow(A[..., i], origin='upper', cmap=plt.cm.Greys)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.axis('Off')
    fig.suptitle(name, va='bottom')
    fig.savefig('%s.png' % name, bbox_inches='tight')

# The flag K.learning_phase() indicates if the network is in training or prediction phase.
# During prediction certain layers such as Dropout are deactivated
inputs = [K.learning_phase()] + model.inputs
conv1_func = K.function(inputs, [conv2.output])
i=1
# plot the activations for test sample i
Xin = B_test[i][np.newaxis]
Xout1 = conv1_func([0] + [Xin])[0][0]
visualize_activation(Xout1, folder+'image%i-conv4' % i)

W1, b1 = conv1.get_weights()
#print W1.shape
#W1=W1.reshape((32,3,3,32))
#print W1.shape
#W1=tf.image.resize_bilinear(W1,(32,32))
#print W1.shape
nx, ny, nc, nf = W1.shape
#print "w1",W1[...,i]
n = np.ceil(nf**.5).astype(int)
fig, axes = plt.subplots(n, n, figsize=(5, 5))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
for i in range(n**2):
    ax = axes.flat[i]
    if i < nf:
        length=len(W1[...,i][0])
        ax.imshow(np.reshape(W1[..., i],(length,length)), origin='upper',cmap=plt.cm.Greys)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.axis('Off')
fig.suptitle('Convolution 1 filter', va='bottom')
fig.savefig(folder+'conv1-filters.png', bbox_inches='tight')


"""
"""
layer_name = 'conv2d_1'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer(layer_name).output)
a=[]
a.append(B_test[i])
a=np.array(a)
intermediate_output = intermediate_layer_model.predict(a)
print intermediate_output.shape

"""

conv1 = model.layers[0]
act1  = model.layers[1]
conv2 = model.layers[4]
act2 = model.layers[5]
avepool=model.layers[6]
"""
W1, b1 = conv1.get_weights()
W2, b2 = conv2.get_weights()
"""



def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"

    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.8, N+4)
    return mycmap

mycmap = transparent_cmap(plt.cm.Reds)


im=[265,0,162,4,3,1,2]

for i in im:   
    print i
    folder = 'task2/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    inputs = [K.learning_phase()] + model.inputs
    conv1_func = K.function(inputs, [conv1.output])
    conv2_func = K.function(inputs, [conv2.output])
    
    class_weights = conv2.get_weights()[0]

    Xin = B_test[i][np.newaxis]
    Xout1 = conv1_func([0] + [Xin])[0][0]
    Xout2 = conv2_func([0] + [Xin])[0][0]
    
    cam = np.zeros(dtype = np.float32, shape = Xout2.shape[1:3])
    target_class = 1
    for j, w in enumerate(class_weights[:, target_class]):
        cam += w * Xout2[j, :, :]
    #print cam.shape  
    cam /= np.max(cam)
    grid = np.resize(cam, (32, 32))

    
    Yp = model.predict(B_test)
    yp = np.argmax(Yp, axis=1)
    """
    dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=folder+'image%i.png' % i)
    """
    w, h = 32, 32
    y, x = np.mgrid[0:h, 0:w]
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(X_test[i],cmap=plt.cm.Greys)
    
    cb = ax.contourf(x, y, grid, 10, cmap=mycmap)
    plt.colorbar(cb)
    plt.savefig(folder+"map"+str(i)+".png")



