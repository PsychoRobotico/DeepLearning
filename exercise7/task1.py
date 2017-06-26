from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import famous
import keras
from keras.layers import *
from keras.regularizers import l2
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.models import load_model

"""
The dataset contains 29900 simulated cosmic ray air events detected with
a FAMOUS-61 telescope at the HAWC observatory (courtesy of Merlin Schaufel).
Note: This is a simplified dataset for demonstration purposes.

Your task is to train a classifier that identifies photons from a background of
proton air showers, based on the following features:
- particle  particle type: 0 = proton, 1 = photon
- pixels    shower image: measured photons for each of the 61 pixels
- logE      log10(E / eV) (measured by HAWC or FAMOUS)
- phi       direction phi [rad] (measured by HAWC)
- theta     direction theta [rad] (measured by HAWC)
- rob       impact point [m] (measured by HAWC)

Sub-tasks:
 a) Set up a classification network to discriminate photons from protons.
    The network should use convolutions to work on the shower image and merge in
    the remaining features at some points in the network.
 b) Investigate the number of photons and protons in the dataset.
    When fitting the model, set the class_weights to give equal importance to
    protons and photons.
 c) Train your network to at least 75% test accuracy.
 d) Investigate the distribution of photon scores.
    What are the precision and recall of your model?
    precision = true positives / (true positives + false positives)
    recall = true positives / (true positives + false negatives)
"""

# ---------------------------------------------------
# Load and prepare dataset - you can leave as is.
# ---------------------------------------------------
data = np.load('/net/scratch/deeplearning/FAMOUS/data.npz')

# shower image: number of detected photons for each pixel (axial coordinates)
S = np.array([famous.vector2matrix(m) for m in data['pixels']])
S = np.log10(S + 1) / 3

# total number of detected photons
logS = np.log10(np.sum(data['pixels'], axis=1))
logS -= logS.min()

# energy - log10(E / 10 TeV)
logE = data['logE'] - 13

# impact point
Rob = data['rob'] / 7000. - 1.5

# shower direction
xdir = np.cos(data['phi'])
ydir = np.sin(data['phi'])

# select features
X1 = S[..., np.newaxis]
X2 = np.stack([logE, Rob, xdir, ydir], axis=-1)

# target features / labels
y = data['particle']  # gamma = 1, proton = 0
print y.shape
print np.count_nonzero(y)

Y = keras.utils.np_utils.to_categorical(y, 2)

# hold out the last 3000 events as test set
X1_train, X1_test = np.split(X1, [-3000])
X2_train, X2_test = np.split(X2, [-3000])
Y_train, Y_test = np.split(Y, [-3000])


# ---------------------------------------------------
# Model & Training
# ---------------------------------------------------
input1 = Input(shape=(9, 9, 1))  # shower image
input2 = Input(shape=(4,))  # other features

# TODO: define a suitable network with a first convolution part working on the
# shower image, and the remaining features merged in later

folder = 'results_task1/'
if not os.path.exists(folder):
    os.makedirs(folder)


def residual_unit(x0):
    x = BatchNormalization (gamma_regularizer=l2(1E-4),
                               beta_regularizer=l2(1E-4))(x0)
    x = Activation("relu")(x)
    x = Convolution2D (16, (2, 2), border_mode="same")(x)
    x = BatchNormalization (gamma_regularizer=l2(1E-4),
                               beta_regularizer=l2(1E-4))(x)
    x = Activation("relu")(x)
    x = Convolution2D (16, (3, 3), border_mode="same")(x)
    return add([x, x0])


z0 = Convolution2D(16, (2, 2),activation="relu")( input1 )
z = residual_unit(z0)
z = MaxPooling2D((2, 2), strides=(1, 1))(z)
z = residual_unit(z)
z = MaxPooling2D((2, 2), strides=(1, 1))(z)
z = residual_unit(z)
z = MaxPooling2D((2, 2), strides=(1, 1))(z)
z = residual_unit(z)
z = BatchNormalization(gamma_regularizer=l2(1E-4),
                               beta_regularizer=l2(1E-4))(z)
z = GlobalAveragePooling2D()(z)
#z = Flatten()(z)

z = concatenate ([z, input2 ])
z = Dense (128,  activation="relu")(z)
z = Dropout(0.1)(z)
z = Dense (64,  activation="relu")(z)
z = Dropout(0.1)(z)
z = Dense (32,  activation="relu")(z)

output = Dense(2, activation='softmax')(z)

model = keras.models.Model(inputs=[input1, input2], outputs=output)

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=1E-3),
    metrics=['accuracy'])

model.fit(
    [X1_train, X2_train],
    Y_train,
    class_weight={0: 1.8, 1: 1.2},  # set class weights to achieve a balanced matching of photons / protons
    batch_size=100,   
    epochs=100,
    verbose=2,
    validation_split=0.1,  # split off 10% training data for validation
    callbacks=[ReduceLROnPlateau(monitor="val_loss", factor =2. / 3, patience=5,verbose =1),
        EarlyStopping(monitor="val_loss", patience =15)])

model.save('model_task1.h5')

#model=load_model("model_task1.h5")
# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------
loss, acc = model.evaluate([X1_test, X2_test], Y_test)
print('Test loss %.3f, test accuracy %.3f' % (loss, acc))


prediction = model.predict([X1_test, X2_test],batch_size=100)
print y[0]
print Y[0]
print prediction.shape
print Y_test.shape
print Y_test[0]
print prediction[0]
#photon=signal=1=(0,1)
true_pos=0
false_pos=0
true_neg=0
false_neg=0


for ind in range(Y_test.shape[0]):
    if (round(prediction[ind][1])==1) and (Y_test[ind][1]==1):
        true_pos+=1
    elif (round(prediction[ind][1])==1) and (Y_test[ind][1]==0):
        false_pos+=1
    elif (round(prediction[ind][1])==0) and (Y_test[ind][1]==1):
        false_neg+=1
    else:
        true_neg+=1

precision=true_pos/(true_pos+false_pos)
recall=true_pos/(true_pos+false_neg)

print "precision",precision
print "recall", recall

# plot distribution of photon-scores
Yp = model.predict([X1_test, X2_test])
y_test = np.argmax(Y_test, axis=1)
s0 = Yp[~y_test.astype(bool), 1]  # predicted photon-score for true protons
s1 = Yp[y_test.astype(bool), 1]  # predicted photon-score for true photons
fig, ax = plt.subplots(1)
plt.hist(s0, label='true protons', bins=np.linspace(0, 1, 31), alpha=0.6, normed=True)
plt.hist(s1, label='true photons', bins=np.linspace(0, 1, 31), alpha=0.6, normed=True)
plt.axvline(0.5, color='r', linestyle='--', label='decision boundary')
plt.legend()
plt.grid()
plt.xlabel('Photon score')
plt.ylabel('$p$(score)')
plt.savefig(folder+'score-distribution.png', bbox_inches='tight')
