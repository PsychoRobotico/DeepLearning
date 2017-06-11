import numpy as np
import matplotlib.pyplot as plt
import dlipr
import keras
from keras.layers import *
from keras import backend as K
import os
from keras.models import *
#from model import *
#import cv2


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

Y_train = dlipr.utils.to_onehot(y_train)
Y_test = dlipr.utils.to_onehot(y_test)

B_train=np.reshape(X_train,(22000,32,32,1))
B_test=np.reshape(X_test,(4000,32,32,1))

# ----------------------------------------------------------
# Load your model
# ----------------------------------------------------------
model = keras.models.load_model('model_conv.h5')

print(model.layers)
print model.summary()
conv1 = model.layers[1]
conv2 = model.layers[3]
conv3 = model.layers[5]
conv4 = model.layers[7]

maxPool = model.layers[4]
globalAveragePool = model.layers[8]





print y_train[1]
print Y_train[1]



folder = 'images/'
if not os.path.exists(folder):
    os.makedirs(folder)

for i in range(2):
    a=[]
    a.append(B_test[i])
    a=np.array(a)
    yp=model.predict(a)
    if (round(yp[0][0])==Y_test[i][0]):
        fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(6, 3.2))
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.15, top=0.98, wspace=0.02)
        dlipr.utils.plot_image(data.test_images[i], ax1)
        ax2.barh(np.arange(2), yp[0], align='center',xerr=Y_test[i])
        ax2.set_yticks(np.arange(2))
        ax2.set_yticklabels(["False","True"])
        ax2.set_xlabel('Propability')
        ax2.set(xlim=(0, 1), xlabel='Probability', yticks=[])
        ax2.text(0.05, 0.1, "False", ha='left', va='center')
        ax2.text(0.05, 1.1, "True", ha='left', va='center')
        ax2.text(0.4,1.3,"T>Tc?",va="center")
        plt.savefig(folder+"image%i.png"%i)
"""
def visualize_activation(A, name='conv'):
    nx, ny, nf = A.shape
    n = np.ceil(nf**.5).astype(int)
    fig, axes = plt.subplots(n, n, figsize=(5, 5))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0, wspace=0)
    for i in range(n**2):
        ax = axes.flat[i]
        if i < nf:
            ax.imshow(A[..., i], origin='upper', cmap=plt.cm.Greys)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.axis('Off')
    fig.suptitle(name, va='bottom')
    fig.savefig('%s.png' % name, bbox_inches='tight')

"""
def visualize_cam(model, layer_idx, filter_indices,
                  seed_img, penultimate_layer_idx=None, alpha=0.5):
    if alpha < 0. or alpha > 1.:
        raise ValueError("`alpha` needs to be between [0, 1]")

    filter_indices = utils.listify(filter_indices)
    print("Working on filters: {}".format(pprint.pformat(filter_indices)))

    # Search for the nearest penultimate `Convolutional` or `Pooling` layer.
    if penultimate_layer_idx is None:
        for idx, layer in utils.reverse_enumerate(model.layers[:layer_idx-1]):
            if isinstance(layer, (_Conv, _Pooling1D, _Pooling2D, _Pooling3D)):
                penultimate_layer_idx = idx
                break

    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Convolution` or `Pooling` '
                         'layer for layer_idx: {}'.format(layer_idx))
    assert penultimate_layer_idx < layer_idx

    losses = [
        (ActivationMaximization(model.layers[layer_idx], filter_indices), 1)
    ]

    penultimate_output = model.layers[penultimate_layer_idx].output
    opt = Optimizer(model.input, losses, wrt=penultimate_output, norm_grads=False)
    _, grads, penultimate_output_value = opt.minimize(seed_img, max_iter=1, verbose=False)

    # We are minimizing loss as opposed to maximizing output as with the paper.
    # So, negative gradients here mean that they reduce loss, maximizing class probability.
    grads *= -1

    # For numerical stability. Very small grad values along with small penultimate_output_value can cause
    # w * penultimate_output_value to zero out, even for reasonable fp precision of float32.
    grads = utils.normalize(grads)

    # Average pooling across all feature maps.
    # This captures the importance of feature map (channel) idx to the output
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    # Generate heatmap by computing weight * output over feature maps
    output_dims = utils.get_img_shape(penultimate_output)[2:]
    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights):
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    # The penultimate feature map size is definitely smaller than input image.
    input_dims = utils.get_img_shape(model.input)[2:]
    heatmap = imresize(heatmap, input_dims, interp='bicubic', mode='F')

    # ReLU thresholding, normalize between (0, 1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    # Create jet heatmap.
    heatmap_colored = np.uint8(cm.jet(heatmap)[..., :3] * 255)
    heatmap_colored = np.uint8(seed_img * alpha + heatmap_colored * (1. - alpha))
    return heatmap_colored

heatmaps = []

seed_img = data.test_images[1]
x = np.expand_dims(seed_img, axis=0)
#x = preprocess_input(x)
a=[]
a.append(B_test[1])
a=np.array(a)
pred_class = np.argmax(model.predict(a))

layer_idx=conv4

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
heatmap = visualize_cam(model, layer_idx, [pred_class], seed_img)
heatmaps.append(heatmap)

plt.axis('off')
plt.imshow(heatmaps)
plt.title('Saliency map')
plt.savefig("map.png")








