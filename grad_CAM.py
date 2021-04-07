#CAUTION: For executing this file, you need to use Tensorflow version 2.2.0
#A trained model on an specific dataset is also needed

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import functions
from matplotlib.collections import LineCollection
import os

last_conv_layer_name = "conv1d_10"
classifier_layer_names = [
    "global_average_pooling1d",
    "dense",
]

os.path.realpath('/')

#=====================================================================================
original_path = os.path.realpath('') + '/UCRArchive_2018/'

#The name of dataset you choose the sample from
folder_name = 'GunPoint'

train_samples, train_labels, _ = functions.utils.read_train_data(functions.utils, original_path, folder_name)

#choosing a random sample
array = train_samples[2:3] 

#Here, change the path to the name of your trained model. (instead of "/val_loss_gunpoint.hdf5") 
model = tf.keras.models.load_model(os.path.realpath('') + '/val_loss_gunpoint.hdf5')

array = np.array(array)
array = array.reshape((1,-1,1))
array = array.astype('float32')

#=====================================================================================
#This part of the code is taken from "https://keras.io/examples/vision/grad_cam/" with a little mix and match

def make_gradcam_heatmap(array, model, last_conv_layer_name, classifier_layer_names):
        # First, we create a model that maps the input image to the activations
    # of the last conv layer
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, i] *= pooled_grads[i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap


heatmap = make_gradcam_heatmap(array, model, last_conv_layer_name, classifier_layer_names)

_,shape1,_ = array.shape

x = np.linspace(0, shape1, num = shape1)
y = train_samples[2]
dydx = heatmap.astype('float64')

points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

fig, axs = plt.subplots()

norm = plt.Normalize(dydx.min(), dydx.max())
lc = LineCollection(segments, cmap='viridis', norm=norm)
lc.set_array(dydx)
lc.set_linewidth(2)
line = axs.add_collection(lc)

fig.colorbar(line, ax=axs)

axs.set_xlim(x.min(), x.max())
axs.set_ylim(-1.25, 2.25)
plt.show()
