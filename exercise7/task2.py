import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.regularizers import l2
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

"""
The dataset contains 200,000 simulated cosmic ray air showers measured with a
surface detector of water Cherenkov tanks, such as the Pierre Auger Observatory.
The primary particles are protons with energies between 10^18.5 and 10^20 eV
and uniformly distributed arrival directions with a maximum zenith angle of 60 degrees.

Each detector station measures the following 2 quantities, which are stored in
form of a map (2D array) corresponding to the station positions in offset coordinates
- time: time point of detection in [s]
- signal: signal strength [arbitrary units]

The following shower properties need to be reconstructed
- showeraxis: x,y,z unit vector of the shower arrival direction
- showercore: position of the shower core in [m]
- logE: log10(energy / eV)

Your tasks are
 a) Complete the provided script to set up a multi-task regression network for
    simultaneously predicting all three shower properties. The network should
    consist of a part that is common to all three objectives, followed by an
    individual subnetwork / tower for each objective.
    As objectives you can use mean squared error.
 b) We want to train the network to the following precision (evaluation provided)
    - less than 1.5 degrees angular resolution
    - less than 25 m core position resolution
    - less than 10% relative energy uncertainty: (E_predicted - E_true) / E_true
    Estimate what these requirements mean in terms of mean squared error loss
    and adjust the relative weights of the objectives accordingly.
 c) Train your network to the above precision.
 d) Plot and interpret the training curve, both with and without the objective weights
"""

# ---------------------------------------------------
# Load and prepare dataset - you can leave as is.
# ---------------------------------------------------
data = np.load('/net/scratch/deeplearning/airshower/auger-shower-planar.npz')

# time map, values standard normalized with untriggered stations set to 0
T = data['time']
T -= np.nanmean(T)
T /= np.nanstd(T)
T[np.isnan(T)] = 0

# signal map, values normalized to range 0-1, untriggered stations set to 0
S = data['signal']
S = np.log10(S)
S -= np.nanmin(S)
S /= np.nanmax(S)
S[np.isnan(S)] = 0

# input features
X = np.stack([T, S], axis=-1)

# target features
# direction - x,y,z unit vector
y1 = data['showeraxis']

# core position - x,y [m]
y2 = data['showercore'][:, 0:2]
y2 /= 750

# energy - log10(E/eV) in range [18.5, 20]
y3 = data['logE']
y3 -= 19.25

# hold out the last 20000 events as test data
X_train, X_test = np.split(X, [-20000])
y1_train, y1_test = np.split(y1, [-20000])
y2_train, y2_test = np.split(y2, [-20000])
y3_train, y3_test = np.split(y3, [-20000])


# ----------------------------------------------------------------------
# Model & Training
# ----------------------------------------------------------------------
input1 = Input(shape=(9, 9, 2))

# TODO: define a suitable network consisting of 2 parts:
# 1) a common network part (you can try a convolutional stack with ResNet- or
#    or DenseNet-like shortcuts)
#   z = ...
# 2) separate network parts for the individual objectives
#   z1 = ...
#   z2 = ...
#   z3 = ...
"""
output1 = Dense(3, name='direction')(z1)
output2 = Dense(2, name='core')(z2)
output3 = Dense(1, name='energy')(z3)

model = keras.models.Model(inputs=input1, outputs=[output1, output2, output3])
"""

folder = 'results_task2_weights/'
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
z = MaxPooling2D((2,2),strides=(1,1))(z)
z = residual_unit(z)
z = MaxPooling2D((2,2),strides=(1,1))(z)
z = residual_unit(z)
z = BatchNormalization(gamma_regularizer=l2(1E-4),
                               beta_regularizer=l2(1E-4))(z)
                               
                               
z1=Convolution2D(16,(2,2),border_mode="same")(z)
z1=residual_unit(z)
z1=MaxPooling2D((2,2),strides=(1,1))(z1)
z1=residual_unit(z1)
z1=Flatten()(z1)
output1 = Dense(3, name='direction')(z1)

z2=Convolution2D(16,(2,2),border_mode="same")(z)
z2=residual_unit(z2)
z2=MaxPooling2D((2,2),strides=(1,1))(z2)
z2=residual_unit(z2)
z2=Flatten()(z2)
output2 = Dense(2, name='core')(z2)

z3=Convolution2D(16,(2,2),border_mode="same")(z)
z3=residual_unit(z3)
z3=MaxPooling2D((2,2),strides=(1,1))(z3)
z3=residual_unit(z3)
z3=Flatten()(z3)
output3 = Dense(1, name='energy')(z3)

model = keras.models.Model(inputs=input1, outputs=[output1, output2, output3])

print(model.summary())

model.compile(
    loss=['mse', 'mse', 'mse'],
    loss_weights=[2, 2, 1],  # you can give more weight to individual objectives
    optimizer=keras.optimizers.Adam(lr=1E-3))

fit = model.fit(X_train,
                [y1_train, y2_train, y3_train],
                batch_size=100,    #hyperparameter
                epochs=200,
                verbose=2,
                validation_split=0.1,  # split off 10% training data for validation
                callbacks=[ReduceLROnPlateau(monitor="val_loss", factor =2. / 3, patience=5,verbose =1),
                    EarlyStopping(monitor="val_loss", patience =10)])

model.save('model-task2_weights.h5')


# TODO: plot training history - with and without weights
plt.figure()
plt.clf()
plt.plot(fit.history['direction_loss'])
plt.plot(fit.history["val_direction_loss"])
plt.legend(["training loss","validation loss"])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig(folder+'direction_loss.png')
    
plt.figure()
plt.clf()
plt.plot(fit.history['core_loss'])
plt.plot(fit.history["val_core_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training loss","validation loss"])
plt.savefig(folder+'core_loss.png')

plt.figure()
plt.clf()
plt.plot(fit.history['energy_loss'])
plt.plot(fit.history["val_energy_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training loss","validation loss"])
plt.savefig(folder+'energy_loss.png')


# ----------------------------------------------------------------------
# Evaluation - this should work as is.
# ----------------------------------------------------------------------
losses = model.evaluate(X_test, [y1_test, y2_test, y3_test], batch_size=128, verbose=0)
print('Test loss')
print('%.5e (direction)' % losses[1])
print('%.5e (core)' % losses[2])
print('%.5e (energy)' % losses[3])
print('%.5e (sum)' % losses[0])

# predict output for test set and undo feature scaling
y1p, y2p, y3p = model.predict(X_test, batch_size=128)
y2_test *= 750  # core position
y3_test += 19.25  # energy
y2p *= 750
y3p += 19.25
y3p = y3p[:, 0]  # remove unnecessary last axis

# direction
d = np.sum(y1p * y1_test, axis=1) / np.sum(y1p**2, axis=1)**.5
d = np.arccos(np.clip(d, 0, 1)) * 180 / np.pi
reso = np.percentile(d, 68)
plt.figure()
plt.hist(d, bins=np.linspace(0, 3, 41))
plt.axvline(reso, color='C1')
plt.text(0.95, 0.95, '$\sigma_{68} = %.2f^\circ$' % reso, ha='right', va='top', transform=plt.gca().transAxes)
plt.xlabel(r'$\Delta \alpha$ [deg]')
plt.ylabel('#')
plt.grid()
plt.savefig(folder+'hist-direction.png', bbox_inches='tight')

# core position
d = np.sum((y2_test - y2p)**2, axis=1)**.5
reso = np.percentile(d, 68)
plt.figure()
plt.hist(d, bins=np.linspace(0, 40, 41))
plt.axvline(reso, color='C1')
plt.text(0.95, 0.95, '$\sigma_{68} = %.2f m$' % reso, ha='right', va='top', transform=plt.gca().transAxes)
plt.xlabel('$\Delta r$ [m]')
plt.ylabel('#')
plt.grid()
plt.savefig(folder+'hist-core.png', bbox_inches='tight')

# energy
d = 10**(y3p - y3_test) - 1
reso = np.std(d)
plt.figure()
plt.hist(d, bins=np.linspace(-0.3, 0.3, 41))
plt.xlabel('($E_\mathrm{rec} - E_\mathrm{true}) / E_\mathrm{true}$')
plt.ylabel('#')
plt.text(0.95, 0.95, '$\sigma = %.3f$' % reso, ha='right', va='top', transform=plt.gca().transAxes)
plt.grid()
plt.savefig(folder+'hist-energy.png', bbox_inches='tight')

plt.figure()
plt.scatter(y3_test, y3p)
plt.plot([18.5, 20], [18.5, 20], color='black')
plt.xlabel('$\log_{10}(E_\mathrm{true}/\mathrm{eV})$')
plt.ylabel('$\log_{10}(E_\mathrm{rec}/\mathrm{eV})$')
plt.grid()
plt.savefig(folder+'scat_energy.png', bbox_inches='tight')
