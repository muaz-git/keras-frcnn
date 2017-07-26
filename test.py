from vgg import rpn
from vgg import nn_base
from keras.models import Input, Model
from keras.applications.vgg16 import VGG16

from keras.utils.vis_utils import plot_model
input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)

x = nn_base(img_input)
nn = rpn(x, 2)
model_rpn = Model(img_input, nn[1])
plot_model(model_rpn, 'model.png')
print model_rpn.summary()