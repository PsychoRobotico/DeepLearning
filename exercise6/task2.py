import numpy as np
import keras
import dlipr
import os
import matplotlib.pyplot as plt
# ------------------------------------------------------------
# Load a small dataset of flower photos
# There are ~2600 (training) + 1000 (test) images in 5 categories.
# To work with all networks all images are rescaled to 224 x 224 pixels.
# ------------------------------------------------------------
data = dlipr.flower.load_data()
data.plot_examples(fname='examples.png')

X_train, X_valid = np.split(data.train_images, [-300])
y_train, y_valid = np.split(data.train_labels, [-300])
X_test = data.test_images
y_test = data.test_labels

mX = np.mean(data.train_images, axis=(0, 1, 2))
sX = np.std(data.train_images, axis=(0, 1, 2))
X_train = (X_train - mX) / sX
X_valid = (X_valid - mX) / sX
X_test = (X_test - mX) / sX

Y_train = dlipr.utils.to_onehot(y_train)
Y_valid = dlipr.utils.to_onehot(y_valid)
Y_test = dlipr.utils.to_onehot(y_test)

# ------------------------------------------------------------
# Use a pretrained network to as feature extractor
# See https://keras.io/applications/
# ------------------------------------------------------------
# Example:
# model = keras.applications.Xception(include_top=False)

from keras.applications.resnet50 import ResNet50
#from keras.applications.inception_v3 import InceptionV3
model_base= ResNet50(weights="imagenet",include_top=False)
#model_base= InceptionV3(weights='imagenet',include_top=False)

features_train=model_base.predict(X_train)
features_valid=model_base.predict(X_valid)
features_test=model_base.predict(X_test)

#5*5*2048=

train_x=features_train.reshape(2370,2048)
valid_x=features_valid.reshape(300,2048)
test_x=features_test.reshape(1000,2048)

#print train_x.shape
#print features_valid.shape
#print features_test.shape

# ------------------------------------------------------------
# Train a simple classifier based on the extracted features
# ------------------------------------------------------------
from keras.layers import *
model=keras.models.Sequential()
model.add(Dense(150, input_dim=2048, activation='softmax'))
model.add(Dropout(0.55))
model.add(Dense(5))
model.add(Activation('softmax'))

print model.summary()

model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=1E-3), metrics=['accuracy'])

# fitting the model 
fit=model.fit(train_x, Y_train, epochs=250, batch_size=128,validation_data=(valid_x,Y_valid),verbose=0)

# ------------------------------------------------------------
# Evaluate your classifier
# ------------------------------------------------------------
loss, acc = model.evaluate(test_x, Y_test, verbose=0, batch_size=128)
print('Test performance fully')
print('Loss = %.4f, acc = %.4f' % (loss, acc))

folder = 'task2/'
if not os.path.exists(folder):
    os.makedirs(folder)

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