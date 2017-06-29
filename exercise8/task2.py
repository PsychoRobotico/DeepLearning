"""
Reconstruct the measured scattering patterns with your trained autoencoders.
Compare the results.
"""
import numpy as np
import dlipr
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.regularizers import l2
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model


def preprocess(X, Y):
    X = np.log10(X + 0.01)
    Y = np.log10(Y + 0.01)
    m = np.max(Y, axis=(1, 2), keepdims=True)  # maximum of undistorted signal
    X = np.clip(X / m, 0, 1.1)
    Y = np.clip(Y / m, 0, 1.1)
    return X[..., np.newaxis], Y[..., np.newaxis]


# read in measured scattering patterns
X_measured = np.empty((3, 64, 64))

for i, fname in enumerate(['cb019_100.npy', 'cb019_103.npy', 'AuCd_302_0K_H_III_1.npy']):
    X_measured[i] = np.load(fname).reshape(64, 64)

X_measured, _ = preprocess(X_measured, X_measured)


model_flat=load_model('model_task1_flat.h5')
model_deep=load_model('model_task1_deep.h5')
model_deep_shortcut=load_model('model_task1_deep_shortcurt.h5')


decoded_imgs_flat = model_flat.predict(X_measured)
decoded_imgs_deep = model_deep.predict(X_measured)
decoded_imgs_deep_shortcut = model_deep_shortcut.predict(X_measured)


N=len(X_measured)
plt.figure()
for i in range(N):
    # display original
    ax = plt.subplot(4, N, i + 1)
    plt.imshow(X_measured[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display flat
    ax = plt.subplot(4, N, i + 1 + N)
    plt.imshow(decoded_imgs_flat[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display deep
    ax = plt.subplot(4, N, i + 1 + 2*N)
    plt.imshow(decoded_imgs_deep[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display deep_shortcut
    ax = plt.subplot(4, N, i + 1 + 3*N)
    plt.imshow(decoded_imgs_deep_shortcut[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig("task2.png")



