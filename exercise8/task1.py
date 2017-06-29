"""
Try to remove the speckles from the test images.
Set up models for a flat and a deep autoencoder.
Train the deep autoencoder with and without shortcut connections.
"""
import numpy as np
import dlipr
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.regularizers import l2
import os
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

data = dlipr.speckles.load_data()
data.plot_examples(fname='examples.png')


def preprocess(X, Y):
    X = np.log10(X + 0.01)
    Y = np.log10(Y + 0.01)
    m = np.max(Y, axis=(1, 2), keepdims=True)  # maximum of undistorted signal
    X = np.clip(X / m, 0, 1.1)
    Y = np.clip(Y / m, 0, 1.1)
    return X[..., np.newaxis], Y[..., np.newaxis]


X_train, Y_train = preprocess(data.X_train, data.Y_train)
X_test, Y_test = preprocess(data.X_test, data.Y_test)


# some indices of interesting test data (it is not necessary to inspect all)
n = [1, 2, 4, 6, 10, 12, 14, 16, 22, 25, 27, 28, 30, 32, 35, 37,
     39, 40, 41, 48, 49, 50, 52, 57, 61, 63, 64, 67, 70, 76]

my_n=[1,2,10,41,52]


################################## FLAT autoencoder
folder = 'results_task1_flat/'
if not os.path.exists(folder):
    os.makedirs(folder)
    

input_img = Input(shape=(64, 64, 1))    

encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

decoded = Conv2D(1, (3, 3), activation='relu', padding='same')(encoded)

autoencoder_flat = keras.models.Model(inputs=input_img, outputs=decoded)

print(autoencoder_flat.summary())

autoencoder_flat.compile(optimizer=keras.optimizers.Adam(lr=1E-3), loss='mse')

fit_flat = autoencoder_flat.fit(X_train,
                Y_train,
                batch_size=128, 
                epochs=50,
                verbose=2,
                validation_split=0.1,  # split off 10% training data for validation
                callbacks=[ReduceLROnPlateau(monitor="val_loss", factor =2. / 3, patience=5,verbose =1),
                            keras.callbacks.CSVLogger('history-flat.csv'),
                            EarlyStopping(monitor="val_loss", patience =10)])

autoencoder_flat.save('model_task1_flat.h5')

h = np.genfromtxt('history-flat.csv', delimiter=',', names=True)

plt.figure(0)
plt.clf()
plt.semilogy(fit_flat.history['loss'])
plt.semilogy(fit_flat.history["val_loss"])
plt.legend(["training loss","validation loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(folder+'loss_flat.png')

loss_flat = autoencoder_flat.evaluate(X_test, Y_test, batch_size=128, verbose=0)
print loss_flat


X_2predict_images=[]
Y_2predict_images=[]
for i in my_n:
    X_2predict_images.append(X_test[i])
    Y_2predict_images.append(Y_test[i])
X_2predict_images=np.array(X_2predict_images)
Y_2predict_images=np.array(Y_2predict_images)

decoded_imgs = autoencoder_flat.predict(X_2predict_images)


#plt.figure(figsize=(5, 3))
plt.figure()
for i in range(len(my_n)):
    # display original
    ax = plt.subplot(3, len(my_n), i + 1)
    plt.imshow(X_2predict_images[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display Y_test
    ax = plt.subplot(3, len(my_n), i + 1 + len(my_n))
    plt.imshow(Y_2predict_images[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, len(my_n), i + 1 + 2*len(my_n))
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(folder+"predictions.png")



################################## deep autoencoder
folder = 'results_task1_deep/'
if not os.path.exists(folder):
    os.makedirs(folder)

input_img = Input(shape=(64, 64, 1))    

x1 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(input_img)
x2 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x1)
x3 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x2)
x4 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x3)
x5 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x4)
encoded = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x5)


x = Conv2D(32, (3, 3), padding='same')(encoded)
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(1, (3, 3), padding='same')(x)
decoded=Activation("relu")(x)



autoencoder_deep = keras.models.Model(inputs=input_img, outputs=decoded)

print(autoencoder_deep.summary())

autoencoder_deep.compile(optimizer=keras.optimizers.Adam(lr=1E-3), loss='mse')

fit_deep = autoencoder_deep.fit(X_train,
                Y_train,
                batch_size=128, 
                epochs=50,
                verbose=2,
                validation_split=0.1,  # split off 10% training data for validation
                callbacks=[ReduceLROnPlateau(monitor="val_loss", factor =2. / 3, patience=5,verbose =1),
                            keras.callbacks.CSVLogger('history-deep.csv'),
                            EarlyStopping(monitor="val_loss", patience =10)])

autoencoder_deep.save('model_task1_deep.h5')


h = np.genfromtxt('history-deep.csv', delimiter=',', names=True)

plt.figure(1)
plt.clf()
plt.semilogy(fit_deep.history['loss'])
plt.semilogy(fit_deep.history["val_loss"])
plt.legend(["training loss","validation loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(folder+'loss_deep.png')

loss_deep = autoencoder_deep.evaluate(X_test, Y_test, batch_size=128, verbose=0)
print loss_deep


X_2predict_images=[]
Y_2predict_images=[]
for i in my_n:
    X_2predict_images.append(X_test[i])
    Y_2predict_images.append(Y_test[i])
X_2predict_images=np.array(X_2predict_images)
Y_2predict_images=np.array(Y_2predict_images)

decoded_imgs = autoencoder_deep.predict(X_2predict_images)


#plt.figure(figsize=(5, 3))
plt.figure()
for i in range(len(my_n)):
    # display original
    ax = plt.subplot(3, len(my_n), i + 1)
    plt.imshow(X_2predict_images[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display Y_test
    ax = plt.subplot(3, len(my_n), i + 1 + len(my_n))
    plt.imshow(Y_2predict_images[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, len(my_n), i + 1 + 2*len(my_n))
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(folder+"predictions.png")




################################## deep autoencoder
folder = 'results_task1_deep_shortcurt/'
if not os.path.exists(folder):
    os.makedirs(folder)

input_img = Input(shape=(64, 64, 1))    

x1 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(input_img)
x2 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x1)
x3 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x2)
x4 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x3)
x5 = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x4)
encoded = Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same')(x5)


x = Conv2D(32, (3, 3), padding='same')(encoded)
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Add()([x5,x])
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Add()([x4,x])
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Add()([x3,x])
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Add()([x2,x])
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(32, (3, 3), padding='same')(x)
x = Add()([x1,x])
x = Activation("relu")(x)
x = Conv2DTranspose(32,(2,2),strides =(2, 2),padding="same")(x)

x = Conv2D(1, (3, 3), padding='same')(x)
x = Add()([input_img,x])
decoded=Activation("relu")(x)


autoencoder_deep_shortcurt = keras.models.Model(inputs=input_img, outputs=decoded)

print(autoencoder_deep_shortcurt.summary())

autoencoder_deep_shortcurt.compile(optimizer=keras.optimizers.Adam(lr=1E-3), loss='mse')

fit_deep_shortcut = autoencoder_deep_shortcurt.fit(X_train,
                Y_train,
                batch_size=128, 
                epochs=100,
                verbose=2,
                validation_split=0.1,  # split off 10% training data for validation
                callbacks=[ReduceLROnPlateau(monitor="val_loss", factor =2. / 3, patience=5,verbose =1),
                            keras.callbacks.CSVLogger('history-deep_shortcurt.csv'),
                            EarlyStopping(monitor="val_loss", patience =10)])

autoencoder_deep_shortcurt.save('model_task1_deep_shortcurt.h5')


h = np.genfromtxt('history-deep_shortcurt.csv', delimiter=',', names=True)

plt.figure(2)
plt.clf()
#plt.plot(fit.history['loss'])
plt.semilogy(fit_deep_shortcut.history['loss'])
#plt.plot(fit.history["val_loss"])
plt.semilogy(fit_deep_shortcut.history["val_loss"])
plt.legend(["training loss","validation loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig(folder+'loss_deep_shortcurt.png')

loss_deep_shortcurt = autoencoder_deep_shortcurt.evaluate(X_test, Y_test, batch_size=128, verbose=0)
print loss_deep_shortcurt

X_2predict_images=[]
Y_2predict_images=[]
for i in my_n:
    X_2predict_images.append(X_test[i])
    Y_2predict_images.append(Y_test[i])
X_2predict_images=np.array(X_2predict_images)
Y_2predict_images=np.array(Y_2predict_images)

decoded_imgs = autoencoder_deep_shortcurt.predict(X_2predict_images)


#plt.figure(figsize=(5, 3))
plt.figure()
for i in range(len(my_n)):
    # display original
    ax = plt.subplot(3, len(my_n), i + 1)
    plt.imshow(X_2predict_images[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display Y_test
    ax = plt.subplot(3, len(my_n), i + 1 + len(my_n))
    plt.imshow(Y_2predict_images[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # display reconstruction
    ax = plt.subplot(3, len(my_n), i + 1 + 2*len(my_n))
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig(folder+"predictions.png")







