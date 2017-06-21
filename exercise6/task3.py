import numpy as np
import densenet
import keras
from keras.preprocessing.image import ImageDataGenerator
import dlipr
import matplotlib.pyplot as plt
import os
# -----------------------------------------
# Data
# -----------------------------------------
data = dlipr.cifar.load_cifar10()
# data sets, hold out 4000 training images for validation
X_train, X_valid = np.split(data.train_images, [-4000])
y_train, y_valid = np.split(data.train_labels, [-4000])
X_test = data.test_images
y_test = data.test_labels

# channel-wise standard normalization
mX = np.mean(data.train_images, axis=(0, 1, 2))
sX = np.std(data.train_images, axis=(0, 1, 2))
X_train = (X_train - mX) / sX
X_valid = (X_valid - mX) / sX
X_test = (X_test - mX) / sX

Y_train = dlipr.utils.to_onehot(y_train)
Y_valid = dlipr.utils.to_onehot(y_valid)
Y_test = dlipr.utils.to_onehot(y_test)

# -----------------------------------------
# Train a DenseNet on the CIFAR-10 challenge
# -----------------------------------------
# choose suitable parameters
dense=4 #number of dense blocks (default 3)
layers=25 #number of convolution layers per dense block
growth=12 #number of filters k to add per convolution
filters=20 # initial number of filters (default 16)
compression=0.6 #compression factor of transition blocks (0 - 1)
drop=0.2 #dropout fraction
decay=1e-4 #weight decay
epochs=50

model = densenet.DenseNet(
    input_shape=(32, 32, 3),
    num_classes=10,
    dense=dense,
    layers=layers,
    growth=growth,
    filters=filters,
    compression=compression,
    drop=drop,
    decay=decay)

model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=["accuracy"])

# data augmentation, what does this do?
generator = ImageDataGenerator(
    width_shift_range=4. / 32,
    height_shift_range=4. / 32,
    fill_mode='constant',
    horizontal_flip=True,
    rotation_range=15,
    shear_range=5)

batch_size = 64
steps_per_epoch = len(X_train) // batch_size

# fit using augmented data generator
fit=model.fit_generator(
    generator.flow(X_train, Y_train, batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=(X_valid, Y_valid),  # fit_generator doesn't work with validation_split
    verbose=0,
    callbacks=[])

# -----------------------------------------
# Evaluation
# -----------------------------------------
folder = 'task3_2/'
if not os.path.exists(folder):
    os.makedirs(folder)

Yp = model.predict(X_test)
yp = np.argmax(Yp, axis=1)

# plot the confusion matrix
dlipr.utils.plot_confusion(yp, data.test_labels, data.classes,
                              fname=folder + 'confusion.png')
for i in range(3):
    dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=folder + 'test-%i.png' % i)
    

[loss, accuracy] = model.evaluate(X_test, Y_test, verbose=0)

print ("testloss:", loss)
print ("testaccuracy:", accuracy)

print "parameters:"
print "dense=",dense
print "layers=",layers
print "growth=",growth
print "filters=",filters
print "compression=", compression
print "drop=", drop
print "decay=", decay

#print fit.history["acc"].shape
#print fit.history["val_acc"].input_shape
#print fit.history['loss'].shape
#print fit.history["val_loss"].shape

plt.figure(0)
plt.plot(fit.history['acc'])
plt.plot(fit.history["val_acc"])
plt.legend(["training accuracy","validation accuracy"])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.savefig(folder+'acc.png')

plt.figure(1)
plt.plot(fit.history['loss'])
plt.plot(fit.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training loss","validation loss"])
plt.savefig(folder+'loss.png')