import keras.backend as K
from keras.layers import Activation, Conv2D, Dense, Flatten, GlobalAveragePooling2D, Input, MaxPooling2D, Reshape, ZeroPadding2D
from keras.layers.merge import concatenate
from keras.models import Model

from ssd_layers import Normalize, PriorBox


def SSD(input_shape, num_classes=21):
    img_size = (input_shape[1], input_shape[0])
    input0 = Input(shape=input_shape)
    # Block 1
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(input0)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)
    # Block 2
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(pool1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(conv2_1)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)
    # Block 3
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(pool2)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(conv3_1)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)
    # Block 4
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(pool3)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(conv4_1)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(conv4_2)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)
    # Block 5
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(pool4)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(conv5_1)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(conv5_2)
    pool5 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)
    # FC6
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', name='fc6')(pool5)
    # fc6 = Dropout(0.5, name='drop6')(fc6)
    # FC7
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='fc7')(fc6)
    # fc7 = Dropout(0.5, name='drop7')(fc7)
    # Block 6
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='conv6_1')(fc7)
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv6_2')(conv6_1)
    # Block 7
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv7_1')(conv6_2)
    conv7_2 = ZeroPadding2D()(conv7_1)
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', name='conv7_2')(conv7_2)
    # Block 8
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name='conv8_1')(conv7_2)
    conv8_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same', name='conv8_2')(conv8_1)
    # Last Pool
    pool6 = GlobalAveragePooling2D(name='pool6')(conv8_2)

    # Prediction from conv4_3
    conv4_3_norm = Normalize(num_classes-1, name='conv4_3_norm')(conv4_3)
    num_priors = 3
    conv4_3_norm_mbox_loc = Conv2D(num_priors*4, (3, 3), padding='same', name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    conv4_3_norm_mbox_loc_flat = Flatten(name='conv4_3_norm_mbox_loc_flat')(conv4_3_norm_mbox_loc)
    name = 'conv4_3_norm_mbox_conf'
    conv4_3_norm_mbox_conf = Conv2D(num_priors*num_classes, (3, 3), padding='same', name=name)(conv4_3_norm)
    conv4_3_norm_mbox_conf_flat = Flatten(name='conv4_3_norm_mbox_conf_flat')(conv4_3_norm_mbox_conf)
    conv4_3_norm_mbox_priorbox = PriorBox(img_size, 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')(conv4_3_norm)
    # Prediction from fc7
    num_priors = 6
    fc7_mbox_loc = Conv2D(num_priors*4, (3, 3), padding='same', name='fc7_mbox_loc')(fc7)
    fc7_mbox_loc_flat = Flatten(name='fc7_mbox_loc_flat')(fc7_mbox_loc)
    name = 'fc7_mbox_conf'
    fc7_mbox_conf = Conv2D(num_priors*num_classes, (3, 3), padding='same', name=name)(fc7)
    fc7_mbox_conf_flat = Flatten(name='fc7_mbox_conf_flat')(fc7_mbox_conf)
    fc7_mbox_priorbox = PriorBox(img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')(fc7)
    # Prediction from conv6_2
    num_priors = 6
    conv6_2_mbox_loc = Conv2D(num_priors*4, (3, 3), padding='same', name='conv6_2_mbox_loc')(conv6_2)
    conv6_2_mbox_loc_flat = Flatten(name='conv6_2_mbox_loc_flat')(conv6_2_mbox_loc)
    name = 'conv6_2_mbox_conf'
    conv6_2_mbox_conf = Conv2D(num_priors*num_classes, (3, 3), padding='same', name=name)(conv6_2)
    conv6_2_mbox_conf_flat = Flatten(name='conv6_2_mbox_conf_flat')(conv6_2_mbox_conf)
    conv6_2_mbox_priorbox = PriorBox(img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')(conv6_2)
    # Prediction from conv7_2
    num_priors = 6
    conv7_2_mbox_loc = Conv2D(num_priors*4, (3, 3), padding='same', name='conv7_2_mbox_loc')(conv7_2)
    conv7_2_mbox_loc_flat = Flatten(name='conv7_2_mbox_loc_flat')(conv7_2_mbox_loc)
    name = 'conv7_2_mbox_conf'
    conv7_2_mbox_conf = Conv2D(num_priors*num_classes, (3, 3), padding='same', name=name)(conv7_2)
    conv7_2_mbox_conf_flat = Flatten(name='conv7_2_mbox_conf_flat')(conv7_2_mbox_conf)
    conv7_2_mbox_priorbox = PriorBox(img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')(conv7_2)
    # Prediction from conv8_2
    num_priors = 6
    conv8_2_mbox_loc = Conv2D(num_priors*4, (3, 3), padding='same', name='conv8_2_mbox_loc')(conv8_2)
    conv8_2_mbox_loc_flat = Flatten(name='conv8_2_mbox_loc_flat')(conv8_2_mbox_loc)
    name = 'conv8_2_mbox_conf'
    conv8_2_mbox_conf = Conv2D(num_priors*num_classes, (3, 3), padding='same', name=name)(conv8_2)
    conv8_2_mbox_conf_flat = Flatten(name='conv8_2_mbox_conf_flat')(conv8_2_mbox_conf)
    conv8_2_mbox_priorbox = PriorBox(img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')(conv8_2)
    # Prediction from pool6
    num_priors = 6
    pool6_mbox_loc_flat = Dense(num_priors*4, name='pool6_mbox_loc_flat')(pool6)
    name = 'pool6_mbox_conf_flat'
    pool6_mbox_conf_flat = Dense(num_priors*num_classes, name=name)(pool6)
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    pool6_reshaped = Reshape(target_shape, name='pool6_reshaped')(pool6)
    pool6_mbox_priorbox = PriorBox(img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')(pool6_reshaped)
    # Gather all predictions
    mbox_loc = concatenate([conv4_3_norm_mbox_loc_flat, fc7_mbox_loc_flat, conv6_2_mbox_loc_flat, conv7_2_mbox_loc_flat, conv8_2_mbox_loc_flat, pool6_mbox_loc_flat], axis=1, name='mbox_loc')
    mbox_conf = concatenate([conv4_3_norm_mbox_conf_flat, fc7_mbox_conf_flat, conv6_2_mbox_conf_flat, conv7_2_mbox_conf_flat, conv8_2_mbox_conf_flat, pool6_mbox_conf_flat], axis=1, name='mbox_conf')
    mbox_priorbox = concatenate([conv4_3_norm_mbox_priorbox, fc7_mbox_priorbox, conv6_2_mbox_priorbox, conv7_2_mbox_priorbox, conv8_2_mbox_priorbox, pool6_mbox_priorbox], axis=1, name='mbox_priorbox')
    if hasattr(mbox_loc, '_keras_shape'):
        num_boxes = mbox_loc._keras_shape[-1]//4
    elif hasattr(mbox_loc, 'int_shape'):
        num_boxes = K.int_shape(mbox_loc)[-1]//4
    mbox_loc = Reshape((num_boxes, 4), name='mbox_loc_final')(mbox_loc)
    mbox_conf = Reshape((num_boxes, num_classes), name='mbox_conf_logits')(mbox_conf)
    mbox_conf = Activation('softmax', name='mbox_conf_final')(mbox_conf)
    predictions = concatenate([mbox_loc, mbox_conf, mbox_priorbox], axis=2, name='predictions')

    model = Model(input0, predictions)

    return model


# def main():
#     model = SSD((80,80,1), 2)
#     model.summary()
# #     model.save('pplcnt_ssd.h5')
#
#
# main()


# import tensorflow.keras as k
#
#
# def SSD(input_shape, num_classes=21):
#     input = k.Input(shape=(80, 80, 1), batch_size=1)
#
#     c1_1 = k.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input)
#     c1_2 = k.layers.DepthwiseConv2D(kernel_size=32, strides=(1, 1), depth_multiplier=1, padding='same', activation='relu')(c1_1)
#     c1_3 = k.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='same')(c1_2)
#
#     c2_1 = k.layers.Conv2D(filters=144, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c1_3)
#     c2_2 = k.layers.DepthwiseConv2D(kernel_size=144, strides=(1, 1), depth_multiplier=1, padding='same', activation='relu')(c2_1)
#     c2_3 = k.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c2_2)
#
#     c3_1 = k.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c2_3)
#     c3_2 = k.layers.DepthwiseConv2D(kernel_size=192, strides=(1, 1), depth_multiplier=1, padding='same', activation='relu')(c3_1)
#     c3_3 = k.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c3_2)
#     c3_4 = k.layers.Add()([c3_3, c2_3])
#
#     c4_1 = k.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c3_4)
#     c4_2 = k.layers.DepthwiseConv2D(kernel_size=192, strides=(1, 1), depth_multiplier=1, padding='same', activation='relu')(c4_1)
#     c4_3 = k.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c4_2)
#     c4_4 = k.layers.Add()([c4_3, c3_4])
#
#     c5_1 = k.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c4_4)
#     c5_2 = k.layers.DepthwiseConv2D(kernel_size=192, strides=(2, 2), depth_multiplier=1, padding='same', activation='relu')(c5_1)
#     c5_3 = k.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(c5_2)
#
#     c6_1 = k.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c5_3)
#     c6_2 = k.layers.DepthwiseConv2D(kernel_size=384, strides=(1, 1), depth_multiplier=1, padding='same', activation='relu')(c6_1)
#     c6_3 = k.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(c6_2)
#     c6_4 = k.layers.Add()([c6_3, c5_3])
#
#     c7_1 = k.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c6_4)
#     c7_2 = k.layers.DepthwiseConv2D(kernel_size=384, strides=(1, 1), depth_multiplier=1, padding='same', activation='relu')(c7_1)
#     c7_3 = k.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(c7_2)
#     c7_4 = k.layers.Add()([c7_3, c6_4])
#
#     c8_1 = k.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c7_4)
#     c8_2 = k.layers.DepthwiseConv2D(kernel_size=384, strides=(1, 1), depth_multiplier=1, padding='same', activation='relu')(c8_1)
#     c8_3 = k.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same')(c8_2)
#     c8_4 = k.layers.Conv2D(filters=576, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c8_3)
#
#     c9_1 = k.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c8_4)
#     c9_2 = k.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(c9_1)
#     c9_3 = k.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(c9_2)
#     c9_4 = k.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(c9_3)
#
#     l1 = k.layers.Conv2D(filters=12, kernel_size=(1, 1), strides=(1, 1), padding='same')(c5_1)
#     l11 = k.layers.Reshape((4800, 1, 4))(l1)
#     l2 = k.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c8_4)
#     l22 = k.layers.Reshape((3200, 1, 4))(l2)
#     l3 = k.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c9_2)
#     l33 = k.layers.Reshape((800, 1, 4))(l3)
#     l4 = k.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(c9_4)
#     l44 = k.layers.Reshape((200, 1, 4))(l4)
#
#     y = k.layers.Concatenate(axis=1)([l11, l22, l33, l44])
#     a = k.layers.Reshape((9000, 4))(y)
#
#     l5 = k.layers.Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='same')(c5_1)
#     l55 = k.layers.Reshape((4800, 2))(l5)
#     l6 = k.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(c8_4)
#     l66 = k.layers.Reshape((3200, 2))(l6)
#     l7 = k.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(c9_2)
#     l77 = k.layers.Reshape((800, 2))(l7)
#     l8 = k.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(c9_4)
#     l88 = k.layers.Reshape((200, 2))(l8)
#
#     y0 = k.layers.Concatenate(axis=1)([l55, l66, l77, l88])
#     r = k.activations.sigmoid(y0)
#
#     # aa = tf.
#
#     output = k.layers.Concatenate()([a, r])
#
#     model = k.Model(input, output)
#
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#
#     return model