import numpy as np
import matplotlib.pyplot as plt
import dlipr
import keras
from keras.layers import *
from keras import backend as K
from keras.utils.np_utils import to_categorical
import os


# ----------------------------------------------------------
# Load your model
# ----------------------------------------------------------
model = keras.models.load_model('model.h5')

print(model.summary())
print(model.layers)

# Note: You need to pick the right convolutional layers from your network here
conv1 = model.layers[0]
conv2 = model.layers[1]



# ----------------------------------------------------------
# Plot the convolutional filters in the first layer
# ----------------------------------------------------------
W1 = conv1.get_weights()[0]

fig, axes = plt.subplots(7, 7, figsize=(3, 3))
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
for i in range(1,49):
    ax = axes.flat[i]
    ax.imshow(255.*W1[:, :,:, i-1], origin='upper')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
axes.flat[0].xaxis.set_visible(False)
axes.flat[0].yaxis.set_visible(False)
axes.flat[0].axis("Off")
fig.suptitle("conv1_test", va='bottom')
fig.savefig('%s.png' % "conv1_test", bbox_inches='tight')



# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
data = dlipr.cifar.load_cifar10()

# prepare the test set the same way as in your training
#X_test = ...

X_test=data.test_images
Y_test=data.test_labels
X_test=X_test.astype("float")/255.
Y_test = to_categorical(Y_test, 10)

num_classes=Y_test.shape[1]


i = 12  # choose a good test sample


# ----------------------------------------------------------
# Plot the picture with predictions
# ----------------------------------------------------------
Xin__ = X_test[i][np.newaxis]
Yp = model.predict(Xin__)
yp = np.argmax(Yp, axis=1)

for k in range(1):
    dlipr.utils.plot_prediction(
        Yp[k],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname='picture_test-%i.png' % i)



# ----------------------------------------------------------
# Plot the activations in convolution layers
# ----------------------------------------------------------
conv1 = model.layers[1]
conv2 = model.layers[3]

def visualize_activation(A, name='conv'):
    nx, ny, nf = A.shape
    n = np.ceil(nf**.5).astype(int)
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
    for i in range(n**2):
        ax = axes.flat[i]
        if i < nf:
            ax.imshow(255.*A[:, :, i], origin='upper', cmap=plt.cm.Greys)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.axis('Off')
    fig.suptitle(name, va='bottom')
    fig.savefig('%s.png' % name, bbox_inches='tight')


# Define a function to call the output of a give layer
# Note: The flag K.learning_phase() indicates if the network is in training or prediction phase.
# During prediction certain layers such as Dropout are deactivated.
inputs = [K.learning_phase()] + model.inputs
conv1_func = K.function(inputs, [conv1.output])
conv2_func = K.function(inputs, [conv2.output])
#conv3_func = K.function(inputs, [conv3.output])

# Get the activations for test image i
Xin = X_test[i][np.newaxis]
Xout1 = conv1_func([0] + [Xin])[0][0]
Xout2 = conv2_func([0] + [Xin])[0][0]
#Xout3 = conv3_func([0] + [Xin])[0][0]

# Note: Using TensorFlow you would do this with
# Xout1 = sess.run(conv1, feed_dict={X:...})

# plot the activations for test sample i
visualize_activation(Xout1, 'image%i-conv1' % i)
visualize_activation(Xout2, 'image%i-conv2' % i)
#visualize_activation(Xout3, 'image%i-conv3' % i)

