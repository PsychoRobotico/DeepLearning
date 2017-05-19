"""
---------------------------------------------------
Exercise 2 - Hypersphere
---------------------------------------------------
In this classification task the data consists of 4D vectors (x1, x2, x3, x4) uniformly sampled in each dimension between (-1, +1).
The data samples are classified according to their 2-norm as inside a hypersphere (|x|^2 < R) or outside (|x|^2 > R).
The task is to train a network to learn this classification based on a relatively small and noisy data set.
For monitoring the training and validating the trained model, we are going to split the dataset into 3 equal parts.
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
np.random.seed(1337)  # for reproducibility

n = 600  # number of data samples
#n = 6000  # number of data samples for task 2.4)
nb_dim = 4  # number of dimensions
R = 1.1  # radius of hypersphere
xdata = 2 * np.random.rand(n, nb_dim) - 1  # features
ydata = np.sum(xdata**2, axis=1)**.5 < R  # labels, True if |x|^2 < R^2, else False

# add some normal distributed noise with sigma = 0.1
xdata += 0.1 * np.random.randn(n, nb_dim)

# turn class labels into one-hot encodings
# 0 --> (1, 0), outside of sphere
# 1 --> (0, 1), inside sphere
y1h = np.stack([~ydata, ydata], axis=-1)

# split data into training, validation and test sets of equal size
n_split = n // 3  # 1/3 of the data
X_train, X_valid, X_test = np.split(xdata, [n_split, 2 * n_split])
y_train, y_valid, y_test = np.split(y1h, [n_split, 2 * n_split])

print("  Training set, shape =", np.shape(X_train), np.shape(y_train))
print("Validation set, shape =", np.shape(X_valid), np.shape(y_valid))
print("      Test set, shape =", np.shape(X_test), np.shape(y_test))


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
#
# TODO: Specify a network with 4 hidden layers of 10 neurons each (ReLU)
# and an output layer (how many nodes?) with softmax activation.
#
model = Sequential([
    Dense(10, activation='relu', input_dim=4), 
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(10, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='SGD',
    metrics=['accuracy'])


earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0., patience=500, verbose=0, mode='auto')

fit = model.fit(
    X_train, y_train,    # training data
    batch_size=n_split,  # no mini-batches, see next lecture
    nb_epoch=8000,       # number of training epochs (normal 8000)
    verbose=2,           # verbosity of shell output (0: none, 1: high, 2: low)
    validation_data=(X_valid, y_valid),  # validation data
    callbacks=[earlyStopping])        # optional list of functions to be called once per epoch

# print a summary of the network layout
print(model.summary())


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

#
# TODO: Obtain training, validation and test accuracy.
# You can use [loss, accuracy] = model.evaluate(X, y, verbose=0)
# Compare with the exact values using your knowledge of the Monte Carlo truth.
# (due to noise the exact accuracy will be smaller than 1)
#
#test values
[loss, accuracy] = model.evaluate(X_test, y_test, verbose=0)
print ("test loss:")
print (loss)
print ("test accuracy:")
print (accuracy)
print ("validation loss:")
print (fit.history["val_loss"][-1])
print ("validation accuracy:")
print (fit.history["val_acc"][-1])
print ("training loss:")
print (fit.history["loss"][-1])
print ("training accuracy:")
print (fit.history["acc"][-1])








#
# TODO: Plot training history in terms of loss and accuracy
# You can obtain these values from the fit.history dictionary.
#
print(fit.history.keys())
print(fit.history['acc'])

f=plt.figure()
plt.plot(fit.history["acc"])
plt.plot(fit.history["val_acc"])
plt.xlabel("epochs")
plt.ylabel("acc")
plt.legend(["training accuracy","validation accuracy"],loc="best")
f.savefig("acc.png")

best_point=(np.argmin(fit.history["val_loss"])+1)
print ("best stopping point due to minimum in plots")
print (best_point) # best stopping point

f=plt.figure()
plt.plot(fit.history["loss"])
plt.plot(fit.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training loss","validation loss"],loc="best")
f.savefig("loss.png")

#
# TODO: For the last sub-task you can use the EarlyStopping callback
# earlystopping = keras.callbacks.EarlyStopping(patience=1)
#