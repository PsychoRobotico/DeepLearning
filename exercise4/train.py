import numpy as np
import matplotlib.pyplot as plt
import dlipr
import keras

import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.layers import Flatten
from keras.layers.convolutional import MaxPooling2D

# ---------------------------------------------------------
# Load and preprocess data
# ---------------------------------------------------------
data = dlipr.cifar.load_cifar10()

# plot some example images
dlipr.utils.plot_examples(data, fname='examples.png')

print(data.train_images.shape)
print(data.train_labels.shape)
print(data.test_images.shape)
print(data.test_labels.shape)

# preprocess the data in a suitable way
classes=data.classes
x_train=data.train_images
y_train=data.train_labels
x_test=data.test_images
y_test=data.test_labels

x_train=x_train.astype("float")/255.
x_test=x_test.astype("float")/255.
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

num_classes=y_test.shape[1]
# ----------------------------------------------------------
# Model and training
# ----------------------------------------------------------

model=Sequential()
model.add(Convolution2D(48, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(48, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(96, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(96, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Convolution2D(192, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Convolution2D(192, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0., patience=500, verbose=0, mode='auto')

print model.summary()

model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(lr=0.0001),
            metrics=['accuracy'])        
       
fit = model.fit(
            x_train, y_train,
            batch_size=128,#80
            epochs=200,
            verbose=1,
            validation_split=0.1,  # split off 10% training data for validation
            callbacks=[earlyStopping])        
        
f=plt.figure()
plt.plot(fit.history["loss"])
plt.plot(fit.history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend(["training loss","validation loss"],loc="best")
f.savefig("loss.png")

f=plt.figure()
plt.plot(fit.history["acc"])
plt.plot(fit.history["val_acc"])
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend(["training accuracy","validation accuracy"],loc="best")
f.savefig("accuracy.png")

       
[loss, accuracy] = model.evaluate(x_test, y_test, verbose=0)
  
val_loss=fit.history["val_loss"][-1]
val_accuracy=fit.history["val_acc"][-1]
 

# predicted probabilities for the test set
Yp = model.predict(x_test)
yp = np.argmax(Yp, axis=1)
 
folder = 'results/'
if not os.path.exists(folder):
    os.makedirs(folder)
 

# plot some test images along with the prediction
for i in range(20):
    dlipr.utils.plot_prediction(
        Yp[i],
        data.test_images[i],
        data.test_labels[i],
        data.classes,
        fname=folder + 'test-%i.png' % i)

# plot the confusion matrix
dlipr.utils.plot_confusion(yp, data.test_labels, data.classes,
                           fname=folder + 'confusion.png')

print ("testloss:", loss)
print ("testaccuracy:", accuracy)
print ("val loss:", val_loss)
print ("val accuracy:", val_accuracy)
  
model.save('model.h5')        