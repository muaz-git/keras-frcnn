import sys
import pickle
import numpy as np
import cv2
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import time


class test_FasterRCNN:
    def __init__(self):
        self.C = None
        self.class_mapping = None
        self.class_to_color = None
        self.num_rois = 32
        self.num_features = None

        self.model_rpn = None
        self.model_classifier_only = None

        self.bbox_threshold = 0.8

        self.setup()

        self.defineNetwork()

    def setup(self):
        sys.setrecursionlimit(40000)
        config_output_filename = 'config.pickle'

        with open(config_output_filename, 'rb') as f_in:
            self.C = pickle.load(f_in)

        # turn off any data augmentation at test time
        self.C.use_horizontal_flips = False
        self.C.use_vertical_flips = False
        self.C.rot_90 = False

        self.class_mapping = self.C.class_mapping

        if 'bg' not in self.class_mapping:
            self.class_mapping['bg'] = len(self.class_mapping)

        self.class_mapping = {v: k for k, v in self.class_mapping.items()}
        print(self.class_mapping)
        self.class_to_color = {self.class_mapping[v]: np.random.randint(0, 255, 3) for v in self.class_mapping}
        self.C.num_rois = int(self.num_rois)

        if self.C.network == 'resnet50':
            self.num_features = 1024
        elif self.C.network == 'vgg':
            self.num_features = 512

    def defineNetwork(self):
        if self.C.network == 'resnet50':
            import keras_frcnn.resnet as nn
        elif self.C.network == 'vgg':
            import keras_frcnn.vgg as nn

        if K.image_dim_ordering() == 'th':
            input_shape_img = (3, None, None)
            input_shape_features = (self.num_features, None, None)
        else:
            input_shape_img = (None, None, 3)
            input_shape_features = (None, None, self.num_features)

        img_input = Input(shape=input_shape_img)
        roi_input = Input(shape=(self.C.num_rois, 4))
        feature_map_input = Input(shape=input_shape_features)

        # define the base network (resnet here, can be VGG, Inception, etc)
        shared_layers = nn.nn_base(img_input, trainable=True)

        # define the RPN, built on the base layers
        num_anchors = len(self.C.anchor_box_scales) * len(self.C.anchor_box_ratios)
        rpn_layers = nn.rpn(shared_layers, num_anchors)

        classifier = nn.classifier(feature_map_input, roi_input, self.C.num_rois, nb_classes=len(self.class_mapping),
                                   trainable=True)

        self.model_rpn = Model(img_input, rpn_layers)
        self.model_classifier_only = Model([feature_map_input, roi_input], classifier)

        model_classifier = Model([feature_map_input, roi_input], classifier)

        print('Loading weights from {}'.format(self.C.model_path))
        self.model_rpn.load_weights(self.C.model_path, by_name=True)
        model_classifier.load_weights(self.C.model_path, by_name=True)

        self.model_rpn.compile(optimizer='sgd', loss='mse')
        model_classifier.compile(optimizer='sgd', loss='mse')

    @staticmethod
    def format_img_size(img, C):
        """ formats the image size based on config """
        img_min_side = float(C.im_size)
        (height, width, _) = img.shape

        if width <= height:
            ratio = img_min_side / width
            new_height = int(ratio * height)
            new_width = int(img_min_side)
        else:
            ratio = img_min_side / height
            new_width = int(ratio * width)
            new_height = int(img_min_side)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return img, ratio

    @staticmethod
    def format_img_channels(img, C):
        """ formats the image channels based on config """
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img[:, :, 0] -= C.img_channel_mean[0]
        img[:, :, 1] -= C.img_channel_mean[1]
        img[:, :, 2] -= C.img_channel_mean[2]
        img /= C.img_scaling_factor
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def format_img(self, img, C):
        """ formats an image for model prediction based on config """
        img, ratio = self.format_img_size(img, C)
        img = self.format_img_channels(img, C)
        return img, ratio

    # Method to transform the coordinates of the bounding box to its original size
    @staticmethod
    def get_real_coordinates(ratio, x1, y1, x2, y2):

        real_x1 = int(round(x1 // ratio))
        real_y1 = int(round(y1 // ratio))
        real_x2 = int(round(x2 // ratio))
        real_y2 = int(round(y2 // ratio))

        return real_x1, real_y1, real_x2, real_y2

    def getBBoxes(self, img):
        st = time.time()

        X, ratio = self.format_img(img, self.C)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = self.model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, self.C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // self.C.num_rois + 1):
            ROIs = np.expand_dims(R[self.C.num_rois * jk:self.C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // self.C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], self.C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = self.model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < self.bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                # print ii
                # print P_cls[0, ii, :]
                # print np.argmax(P_cls[0, ii, :])
                cls_name = self.class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= self.C.classifier_regr_std[0]
                    ty /= self.C.classifier_regr_std[1]
                    tw /= self.C.classifier_regr_std[2]
                    th /= self.C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [self.C.rpn_stride * x, self.C.rpn_stride * y, self.C.rpn_stride * (x + w), self.C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        return ratio, probs, bboxes