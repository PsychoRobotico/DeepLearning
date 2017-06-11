"""
Exercise 5:

Task 1: Classify the magnetic phases in terms of
- a fully connected layer (FCL)
- a convolutional neural network (CNN)
- the toy model (s. lecture slides)
Plot test accuracy vs. temperature for both networks and for the toy model.

Task 2: Discriminative localization
Pick out two correctly and two wrongly classified images from the CNN.
Look at Exercise 4, task 2 (visualize.py) to extract weights and feature maps from the trained model.
Calculate and plot the class activation maps and compare them with the images in order to see which regions lead to the class decision.

Hand in a printout of your commented code and plots.

If you are interested in the data generation look at MonteCarlo.py.
"""

# Note: if you are having troubles with loading the dlipr library you can
# comment in the following two lines.
# import sys
# sys.path.append("/software/community/dlipr")
import dlipr
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *

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

#------------------------Task1 Toy Model--------

def heaviside(x):
    return 0.5 * (np.sign(x) + 1.)
 
size=X_train.shape[0] 

W2=np.array([[2.,1.,-1.],[-2.,-2.,1.]])

N=1024   
accs=[]
loss=[]
eps=[]
"""
for ep in np.arange(0.,0.8,0.001):
    #epsilon=0.1
    epsilon=ep
    eps.append(ep)

    y_toy=[]
    for i in range(size):
        m=np.mean(X_train[i])
        h1=heaviside(1./(1.+epsilon)*np.array([[m-epsilon],[-m-epsilon],[m+epsilon]]))
        y=heaviside(np.dot(W2,h1))
        y_toy.append(y)
    y_toy=np.array(y_toy)

    # objective:
    obj_toy = np.mean((y_toy[:,1,0] - y_train)**2)
    #print obj_toy
    loss.append(obj_toy)
    # accuracy:
    acc_toy = np.mean(np.round(y_toy[:,1,0]) == y_train)
    #print acc_toy
    accs.append(acc_toy)
#-->epsilon=0.5
plt.figure()
plt.plot(eps,accs,label="acc")
plt.plot(eps,loss,label="loss")
plt.legend(loc="best")
plt.savefig("toy_epsilon.png")
"""

counter=np.zeros(len(T))
abs_counter=np.zeros(len(T))

epsilon=0.5

y_toy=[]
for i in range(len(X_test)):
    m=np.mean(X_test[i])
    h1=heaviside(1./(1.+epsilon)*np.array([[m-epsilon],[-m-epsilon],[m+epsilon]]))
    y=heaviside(np.dot(W2,h1))
    y_toy.append(y)
    abs_counter[T_index_test[i]]+=1
    if(y[1][0]==y_test[i]):
        counter[T_index_test[i]]+=1
y_toy=np.array(y_toy)
acc_toy = np.mean(np.round(y_toy[:,1,0]) == y_test)
print acc_toy

accs_toy=counter/abs_counter

plt.figure()
plt.plot(T,accs_toy,label="accuracy")
plt.xlabel("temperature")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.savefig("toy.png")

#-------------------------fully connected layer------------

Y_train = dlipr.utils.to_onehot(y_train)
Y_test = dlipr.utils.to_onehot(y_test)

model=keras.models.Sequential([
    InputLayer(input_shape=(32, 32)),
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.2),
    Dense(64,activation="relu"),
    Dropout(0.2),
    Dense(32,activation="relu"),
    Dropout(0.1),
    Dense(16),
    Dense(2,activation="softmax")])

print(model.summary())

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=1E-3),
    metrics=['accuracy'])


model.fit(
    X_train, Y_train,
    batch_size=32,
    epochs=15,
    validation_split=0.1,
    verbose=0,
    callbacks=[])

model.save('model_fully.h5')

loss, acc = model.evaluate(X_test, Y_test, verbose=0, batch_size=128)
print('Test performance')
print('Loss = %.4f, acc = %.4f' % (loss, acc))

y_fully=[]
counter=np.zeros(len(T))
abs_counter=np.zeros(len(T))


for i in range(len(X_test)):
    a=[]
    a.append(X_test[i])
    a=np.array(a)
    yp=model.predict(a)
    y_fully.append(yp)
    abs_counter[T_index_test[i]]+=1
    if(round(yp[0][0],0)==Y_test[i][0]):
        counter[T_index_test[i]]+=1


        
y_fully=np.array(y_fully)
acc_fully = np.mean(np.round(y_fully[:,0,0]) == Y_test)
print acc_fully
accs_fully=counter/abs_counter

plt.figure()
plt.plot(T,accs_fully,label="accuracy")
plt.xlabel("temperature")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.savefig("fully.png")



#---------------------convolutional network----------------

Y_train = dlipr.utils.to_onehot(y_train)
Y_test = dlipr.utils.to_onehot(y_test)


B_train=np.reshape(X_train,(22000,32,32,1))
B_test=np.reshape(X_test,(4000,32,32,1))


model2 = keras.models.Sequential([
    InputLayer(input_shape=(32, 32,1)),
    Convolution2D(8, (3,3), activation='relu'),
    Dropout(0.2),
    Convolution2D(16, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Convolution2D(32, (3,3), activation='relu'),
    Dropout(0.2),
    Convolution2D(32, (3,3), activation='relu'),
    GlobalAveragePooling2D(),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16),
    Dense(2, activation='softmax')])
    
print(model2.summary())

model2.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=1E-3),
    metrics=['accuracy'])

model2.fit(
    B_train, Y_train,
    batch_size=32,
    epochs=15,
    validation_split=0.1,
    verbose=0,
    callbacks=[])

model2.save('model_conv.h5')

loss, acc = model2.evaluate(B_test, Y_test, verbose=0, batch_size=128)
print('Test performance')
print('Loss = %.4f, acc = %.4f' % (loss, acc))

y_conv=[]
counter=np.zeros(len(T))
abs_counter=np.zeros(len(T))

for i in range(len(B_test)):
    a=[]
    a.append(B_test[i])
    a=np.array(a)
    yp=model2.predict(a)
    y_conv.append(yp)
    abs_counter[T_index_test[i]]+=1
    if(round(yp[0][0],0)==Y_test[i][0]):
        counter[T_index_test[i]]+=1
    else:
        print "here:",i
        
y_conv=np.array(y_conv)
accs_conv=counter/abs_counter

plt.figure()
plt.plot(T,accs_conv,label="accuracy")
plt.xlabel("temperature")
plt.ylabel("accuracy")
plt.legend(loc="best")
plt.savefig("conv.png")

