from vgg import rpn
from vgg import nn_base
from keras.models import Input, Model
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization

from keras.utils.vis_utils import plot_model

model = VGG16()
plot_model(model, 'model.png')
print model.summary()