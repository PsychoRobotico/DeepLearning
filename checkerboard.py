"""
Exercise 1: Checkerboard task
In this task the data consists of 1000 2D vectors (x1, x2) uniformly sampled in the range (-1, 1).
Each vector is classified as signal y=1 if either x1 or x2 > 0 (but not both), similar to y=XOR(x1, x2)
The task is to train a simple neural network to correctly classify the data.
For now we formulate this problem as regression task with the network predicting a single value y_model and the optimizer minimizing the mean squared error between model and data.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# ------------------------------------------
# Data
# ------------------------------------------
np.random.seed(1337)  # for reproducibility

# x = (x1, x2) with random numbers x1, x2 ~ Uniform(-1, 1)
# y ~ XOR(x1, x2)
N = 1000
xdata = 2 * np.random.rand(N, 2) - 1     # shape = (N, 2)
ydata = (xdata[:, 0] * xdata[:, 1]) < 0  # shape = (N)
ydata = ydata.reshape(N, 1)              # shape = (N, 1)


# ------------------------------------------
# Model
# ------------------------------------------
# placeholders for data
x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

# hidden layer, h1 = ReLU(W1*x + b1)
W1 = tf.Variable(tf.random_normal([2, 8], stddev=1))
b1 = tf.Variable(tf.random_normal([8], stddev=0.1))
h1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# output layer, ym = sigmoid(W2*h1 + b2)
W2 = tf.Variable(tf.random_normal([8, 1], stddev=1))
b2 = tf.Variable(tf.random_normal([1], stddev=0.1))
ym = tf.sigmoid(tf.matmul(h1, W2) + b2)

# objective function and optimizer
objective = tf.reduce_mean((ym - y)**2)  # mean squared error
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(objective)

# accuracy (= fraction of correct predictions)
correct_prediction = tf.equal(y, tf.round(ym))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start a session and initialize the computation graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train the model for 20000 steps
for i in range(20000):
    sess.run(train_step, feed_dict={x: xdata, y: ydata})

    if i % 200 == 0:
        obj, acc = sess.run([objective, accuracy], feed_dict={x: xdata, y: ydata})
        print("step: %i, loss %f, accuracy %f" % (i, obj, acc))


# ------------------------------------------
# Evaluation
# ------------------------------------------
# Compute the model prediction on a grid of 500 x 500 points
n = 500
s = np.linspace(-1, 1, n)
xp = np.array(np.meshgrid(s, s))      # xp.shape = (2, 500, 500), all combinations of s with s
xp = xp.T                             # xp.shape = (500, 500, 2)
xp = xp.reshape((-1, 2))              # xp.shape = (500*500, 2), reshape
yp = sess.run(ym, feed_dict={x: xp})  # yp.shape = (500*500, 1), model prediction
yp = yp.reshape((n, n))               # yp.shape = (500, 500), reshape


# plot model prediction
fig, ax = plt.subplots(1)
im = ax.imshow(yp, extent=(-1, 1, -1, 1), origin='lower',
               vmin=0, vmax=1, cmap=plt.cm.seismic_r, alpha=0.8)
cbar = plt.colorbar(im)
cbar.set_label('$y_\mathrm{m}$')

# plot data
colors = ['blue' if y else 'red' for y in ydata]  # 'blue' for y=1 and 'red' for y=0
ax.scatter(*xdata.T, c=colors, lw=0)

ax.set(xlabel='$x_1$', ylabel='$x_2$', aspect='equal')
ax.grid(True)
fig.savefig('checkerboard.png', bbox_inches='tight')

#TASK 1.3:
W1_, b1_, W2_, b2_ = sess.run([W1, b1, W2, b2])

#hidden layer with numpy:


h1_=np.maximum(0.,np.dot(xdata,W1_)+b1_)

#output layer with numpy

def sigmoid(x):
    return 1./(1. + np.exp(-1.*x))


ym_ =sigmoid(np.dot(h1_, W2_) + b2_)

loss_ = np.mean((ym_-ydata)**2)

print "loss:", loss_
print "tensorflow:", obj













#
# TODO: Verify TensorFlow by computing the network output and loss using pure numpy.
#   You can retrieve the weights and biases using
#   W1_, b1_, W2_, b2_ = sess.run([W1, b1, W2, b2])
#   Bonus: Don't loop over the data.
#

#
# TODO: Change the number of neurons in the hidden layer to 8 (and to 2) and retrain the model.
#

#
# TODO: Add an additional hidden layer of 4 neurons with ReLU as activation and retrain the model.
#
