"""
---------------------------------------------------
Exercise 2 - Classification
---------------------------------------------------
Suppose we want to classify some data (4 samples) into 3 distinct classes: 0, 1, and 2.
We have set up a network with a pre-activation output z in the last layer.
Applying softmax will give the final model output.
input X ---> some network --> z --> y_model = softmax(z)

We quantify the agreement between truth (y) and model using categorical cross-entropy.
J = - sum_i (y_i * log(y_model(x_i))

In the following you are to implement softmax and categorical cross-entropy
and evaluate them values given the values for z.
"""
from __future__ import print_function
import numpy as np
import tensorflow as tf


# Data: 4 samples with the following class labels (input features X irrelevant here)
y_cl = np.array([0, 0, 2, 1])

# output of the last network layer before applying softmax
z = np.array([
    [  4,   5,   1],
    [ -1,  -2,  -3],
    [0.1, 0.2, 0.3],
    [ -1, 100,   1]
    ])



# TensorFlow implementation as reference. Make sure you get the same results!
print('\nTensorFlow ------------------------------ ')
with tf.Session() as sess:
    z_ = tf.constant(z, dtype='float64')
    y_ = tf.placeholder(dtype='float64', shape=(None,3))

    y = np.array([[1., 0., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    print('one-hot encoding of data labels')
    print(y)

    y_model = tf.nn.softmax(z)
    crossentropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_model), reduction_indices=[1]))

    print('softmax(z)')
    print(sess.run(y_model))

    print('cross entropy = %f' % sess.run(crossentropy, feed_dict={y_: y}))


print('\nMy solution ------------------------------ ')
# 1) Write a function that turns any class labels y_cl into one-hot encodings y. (2 points)
#    0 --> (1, 0, 0)
#    1 --> (0, 1, 0)
#    2 --> (0, 0, 1)
#    Make sure that np.shape(y) = (4, 3) for np.shape(y_cl) = (4).

def to_onehot(y_cl, num_classes):
    y = np.zeros((len(y_cl), num_classes))
    for i,entry in zip(range(len(y_cl)),y_cl):
        np.put(y[i],y_cl[i],1.0)
    return y

print('One Hot array:')
print(to_onehot(y_cl,3))



# 2) Write a function that returns the softmax of the input z along the last axis. (2 points)
def softmax(z):
    return np.exp(z)/np.expand_dims(np.sum(np.exp(z), axis=-1),axis=-1)

print('Softmax function:')
print(softmax(z))
# 3) Compute the categorical cross-entropy between data and model (2 points)

def categorical_cross_entropy(y,y_model):
    return np.mean(-1.0*np.sum(y * np.log(y_model)))


# 4) Which classes are predicted by the model (maximum entry). (1 point)
def maximum_entry(x):
    return np.argmax(x,axis=1)

print('Prediction:')
print(maximum_entry(softmax(z)))
# 5) How many samples are correctly classified (accuracy)? (1 point)
def accuracy(y_cl,y_prediction):
    acc= np.sum(y_cl==y_prediction)/float(len(y_cl))
    return acc
    
print('Congratulation you have achieved ', accuracy(y_cl,maximum_entry(softmax(z)))*100, '% accuracy',sep='')