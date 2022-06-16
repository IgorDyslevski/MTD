from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, MaxPool2D, Conv2D, Conv2DTranspose, \
    UpSampling2D, Dense, Activation, Dropout, Flatten, Concatenate
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, \
    binary_crossentropy, categorical_crossentropy
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import cv2


input_shape = (300, 300, 3)
filters = [8, 32, 128, 128, 128, 128, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256]
aspect_ratios = {
    0: [1, 2, 1 / 2],
    1: [1, 2, 3, 1 / 2, 1 / 3],
    2: [1, 2, 3, 1 / 2, 1 / 3],
    3: [1, 2, 3, 1 / 2, 1 / 3],
    4: [1, 2, 1 / 2],
    5: [1, 2, 1 / 2],
}
sizes = {
    0: [38, 38],
    1: [19, 19],
    2: [10, 10],
    3: [5, 5],
    4: [3, 3],
    5: [1, 1],
}
convolution_filters = {
    0: 32,
    1: 96,
    2: 128,
    3: 256,
    4: 256,
    5: 128,
}
mtd_convolution_filters = {
    0: 256,
    1: 384,
}
scales = {
    0: 0.1,
    1: 0.2,
    2: 0.375,
    3: 0.55,
    4: 0.725,
    5: 0.9,
}
detection_classes = ['background', 'head']
landmark_classes = ['background', 'left_eye', 'right_eye', 'nose']


# 300 x 300 x 3
input_layer = Input(shape=(300, 300, 3))
# 300 x 300 x 3
conv1_layer = Conv2D(filters=filters[0], kernel_size=3, strides=1, padding='same')(input_layer)
# 300 x 300 x F
conv2_layer = Conv2D(filters=filters[1], kernel_size=3, strides=1, padding='same')(conv1_layer)
# 300 x 300 x F
relu1_layer = Activation('relu')(conv2_layer)
# 300 x 300 x F
maxpool1_layer = MaxPool2D()(relu1_layer)
# 150 x 150 x F
dropout1_layer = Dropout(0.2)(maxpool1_layer)
# 150 x 150 x F
conv3_layer = Conv2D(filters=filters[2], kernel_size=3, strides=1, padding='same')(dropout1_layer)
# 150 x 150 x F
conv4_layer = Conv2D(filters=filters[3], kernel_size=3, strides=1, padding='same')(conv3_layer)
# 150 x 150 x F
relu2_layer = Activation('relu')(conv4_layer)
# 150 x 150 x F
maxpool2_layer = MaxPool2D()(relu2_layer)
# 75 x 75 x F
dropout2_layer = Dropout(0.2)(maxpool2_layer)
# 75 x 75 x F
conv5_layer = Conv2D(filters=filters[4], kernel_size=3, strides=1, padding='same')(dropout2_layer)
# 75 x 75 x F
conv6_layer = Conv2D(filters=filters[5], kernel_size=3, strides=2, padding='same')(conv5_layer)
# 38 x 38 x F
relu3_layer = Activation('relu')(conv6_layer)
# 38 x 38 x F
maxpool3_layer = MaxPool2D(pool_size=(1, 1))(relu3_layer)
# 38 x 38 x F
dropout3_layer = Dropout(0.2)(maxpool3_layer)
# 38 x 38 x F
conv7_layer = Conv2D(filters=filters[6], kernel_size=3, strides=1, padding='same')(dropout3_layer)
# 38 x 38 x F
conv8_layer = Conv2D(filters=filters[7], kernel_size=3, strides=1, padding='same')(conv7_layer)
# 38 x 38 x F
relu4_layer = Activation('relu')(conv8_layer)
# 38 x 38 x F
maxpool4_layer = MaxPool2D()(relu4_layer)
# 19 x 19 x F
dropout4_layer = Dropout(0.2)(maxpool4_layer)
# 19 x 19 x F
conv9_layer = Conv2D(filters=filters[8], kernel_size=3, strides=1, padding='same')(dropout4_layer)
# 19 x 19 x F
conv10_layer = Conv2D(filters=filters[9], kernel_size=3, strides=2, padding='same')(conv9_layer)
# 10 x 10 x F
relu5_layer = Activation('relu')(conv10_layer)
# 10 x 10 x F
maxpool5_layer = MaxPool2D(pool_size=(1, 1))(relu5_layer)
# 10 x 10 x F
dropout5_layer = Dropout(0.2)(maxpool5_layer)
# 10 x 10 x F
conv11_layer = Conv2D(filters=filters[10], kernel_size=3, strides=1, padding='same')(dropout5_layer)
# 10 x 10 x F
conv12_layer = Conv2D(filters=filters[11], kernel_size=3, strides=1, padding='same')(conv11_layer)
# 10 x 10 x F
relu6_layer = Activation('relu')(conv12_layer)
# 10 x 10 x F
maxpool6_layer = MaxPool2D()(relu6_layer)
# 5 x 5 x F
dropout6_layer = Dropout(0.2)(maxpool6_layer)
# 5 x 5 x F
conv13_layer = Conv2D(filters=filters[12], kernel_size=3, strides=1, padding='same')(dropout6_layer)
# 5 x 5 x F
conv14_layer = Conv2D(filters=filters[13], kernel_size=3, strides=2, padding='same')(conv13_layer)
# 3 x 3 x F
relu7_layer = Activation('relu')(conv14_layer)
# 3 x 3 x F
maxpool7_layer = MaxPool2D(pool_size=(1, 1))(relu7_layer)
# 3 x 3 x F
dropout7_layer = Dropout(0.2)(maxpool7_layer)
# 3 x 3 x F
conv15_layer = Conv2D(filters=filters[14], kernel_size=3, strides=1, padding='same')(dropout7_layer)
# 3 x 3 x F
conv16_layer = Conv2D(filters=filters[15], kernel_size=3, strides=1, padding='same')(conv15_layer)
# 3 x 3 x F
relu8_layer = Activation('relu')(conv16_layer)
# 3 x 3 x F
maxpool8_layer = MaxPool2D()(relu8_layer)
# 1 x 1 x F
dropout8_layer = Dropout(0.2)(maxpool8_layer)
# 1 x 1 x F

# First SSD congress
congress1_conv1_layer = Conv2D(filters=convolution_filters[0], kernel_size=3, strides=1, padding='same')(relu4_layer)
congress1_conv2_layer = Conv2D(filters=len(aspect_ratios[0]) * 2,
                               name='FirstSSDCongressCoordinates', # (count filters = count aspect ratios * 2)
                              kernel_size=3, strides=1, padding='same')(congress1_conv1_layer)
congress1_conv3_layer = Conv2D(filters=len(aspect_ratios[0]) * 2,
                               name='FirstSSDCongressSizes', # (count filters = count aspect ratios * 2)
                              kernel_size=3, strides=1, padding='same')(congress1_conv1_layer)
congress1_conv4_layer = Conv2D(filters=len(aspect_ratios[0]) * len(detection_classes),
                               name='FirstSSDCongressClasses', # (count filters = count aspect ratios * count classes)
                              kernel_size=3, strides=1, padding='same')(congress1_conv1_layer)
congress1_tanh1_layer = Activation('tanh', name='FirstTanh')(congress1_conv2_layer)
congress1_sigmoid1_layer = Activation('sigmoid', name='FirstSigmoid')(congress1_conv3_layer)
congress1_softmax1_layer = Activation('softmax', name='FirstSoftmax')(congress1_conv4_layer)
congress1_output1_layer = Concatenate(name='FirstSSDCongress')([congress1_tanh1_layer,
                                                                  congress1_sigmoid1_layer, congress1_softmax1_layer])

# Second SSD congress
congress2_conv1_layer = Conv2D(filters=convolution_filters[1], kernel_size=3, strides=1, padding='same')(conv9_layer)
congress2_conv2_layer = Conv2D(filters=len(aspect_ratios[1]) * 2, name='SecondSSDCongressCoordinates',
                              kernel_size=3, strides=1, padding='same')(congress2_conv1_layer)
congress2_conv3_layer = Conv2D(filters=len(aspect_ratios[1]) * 2, name='SecondSSDCongressSizes',
                              kernel_size=3, strides=1, padding='same')(congress2_conv1_layer)
congress2_conv4_layer = Conv2D(filters=len(aspect_ratios[1]) * len(detection_classes), name='SecondSSDCongressClasses',
                              kernel_size=3, strides=1, padding='same')(congress2_conv1_layer)
congress2_tanh1_layer = Activation('tanh', name='SecondTanh')(congress2_conv2_layer)
congress2_sigmoid1_layer = Activation('sigmoid', name='SecondSigmoid')(congress2_conv3_layer)
congress2_softmax1_layer = Activation('softmax', name='SecondSoftmax')(congress2_conv4_layer)
congress2_output1_layer = Concatenate(name='SecondSSDCongress')([congress2_tanh1_layer, congress2_sigmoid1_layer, congress2_softmax1_layer])

# Third SSD congress
congress3_conv1_layer = Conv2D(filters=convolution_filters[2], kernel_size=3, strides=1, padding='same')(relu6_layer)
congress3_conv2_layer = Conv2D(filters=len(aspect_ratios[2]) * 2, name='ThirdSSDCongressCoordinates',
                              kernel_size=3, strides=1, padding='same')(congress3_conv1_layer)
congress3_conv3_layer = Conv2D(filters=len(aspect_ratios[2]) * 2, name='ThirdSSDCongressSizes',
                              kernel_size=3, strides=1, padding='same')(congress3_conv1_layer)
congress3_conv4_layer = Conv2D(filters=len(aspect_ratios[2]) * len(detection_classes), name='ThirdSSDCongressClasses',
                              kernel_size=3, strides=1, padding='same')(congress3_conv1_layer)
congress3_tanh1_layer = Activation('tanh', name='ThirdTanh')(congress3_conv2_layer)
congress3_sigmoid1_layer = Activation('sigmoid', name='ThirdSigmoid')(congress3_conv3_layer)
congress3_softmax1_layer = Activation('softmax', name='ThirdSoftmax')(congress3_conv4_layer)
congress3_output1_layer = Concatenate(name='ThirdSSDCongress')([congress3_tanh1_layer, congress3_sigmoid1_layer, congress3_softmax1_layer])

# Fourth SSD congress
congress4_conv1_layer = Conv2D(filters=convolution_filters[3], kernel_size=3, strides=1, padding='same')(conv13_layer)
congress4_conv2_layer = Conv2D(filters=len(aspect_ratios[3]) * 2, name='FourthSSDCongressCoordinates',
                              kernel_size=3, strides=1, padding='same')(congress4_conv1_layer)
congress4_conv3_layer = Conv2D(filters=len(aspect_ratios[3]) * 2, name='FourthSSDCongressSizes',
                              kernel_size=3, strides=1, padding='same')(congress4_conv1_layer)
congress4_conv4_layer = Conv2D(filters=len(aspect_ratios[3]) * len(detection_classes), name='FourthSSDCongressClasses',
                              kernel_size=3, strides=1, padding='same')(congress4_conv1_layer)
congress4_tanh1_layer = Activation('tanh', name='FourthTanh')(congress4_conv2_layer)
congress4_sigmoid1_layer = Activation('sigmoid', name='FourthSigmoid')(congress4_conv3_layer)
congress4_softmax1_layer = Activation('softmax', name='FourthSoftmax')(congress4_conv4_layer)
congress4_output1_layer = Concatenate(name='FourthSSDCongress')([congress4_tanh1_layer, congress4_sigmoid1_layer, congress4_softmax1_layer])

# Fifth SSD congress
congress5_conv1_layer = Conv2D(filters=convolution_filters[4], kernel_size=3, strides=1, padding='same')(relu8_layer)
congress5_conv2_layer = Conv2D(filters=len(aspect_ratios[4]) * 2, name='FifthSSDCongressCoordinates',
                              kernel_size=3, strides=1, padding='same')(congress5_conv1_layer)
congress5_conv3_layer = Conv2D(filters=len(aspect_ratios[4]) * 2, name='FifthSSDCongressSizes',
                              kernel_size=3, strides=1, padding='same')(congress5_conv1_layer)
congress5_conv4_layer = Conv2D(filters=len(aspect_ratios[4]) * len(detection_classes), name='FifthSSDCongressClasses',
                              kernel_size=3, strides=1, padding='same')(congress5_conv1_layer)
congress5_tanh1_layer = Activation('tanh', name='FifthTanh')(congress5_conv2_layer)
congress5_sigmoid1_layer = Activation('sigmoid', name='FifthSigmoid')(congress5_conv3_layer)
congress5_softmax1_layer = Activation('softmax', name='FifthSoftmax')(congress5_conv4_layer)
congress5_output1_layer = Concatenate(name='FifthSSDCongress')([congress5_tanh1_layer, congress5_sigmoid1_layer, congress5_softmax1_layer])

# Sixth SSD congress
congress6_conv1_layer = Conv2D(filters=convolution_filters[5], kernel_size=3, strides=1, padding='same')(dropout8_layer)
congress6_conv2_layer = Conv2D(filters=len(aspect_ratios[5]) * 2, name='SixthSSDCongressCoordinates',
                              kernel_size=3, strides=1, padding='same')(congress6_conv1_layer)
congress6_conv3_layer = Conv2D(filters=len(aspect_ratios[5]) * 2, name='SixthSSDCongressSizes',
                              kernel_size=3, strides=1, padding='same')(congress6_conv1_layer)
congress6_conv4_layer = Conv2D(filters=len(aspect_ratios[5]) * len(detection_classes), name='SixthSSDCongressClasses',
                              kernel_size=3, strides=1, padding='same')(congress6_conv1_layer)
congress6_tanh1_layer = Activation('tanh', name='SixthTanh')(congress6_conv2_layer)
congress6_sigmoid1_layer = Activation('sigmoid', name='SixthSigmoid')(congress6_conv3_layer)
congress6_softmax1_layer = Activation('softmax', name='SixthSoftmax')(congress6_conv4_layer)
congress6_output1_layer = Concatenate(name='SixthSSDCongress')([congress6_tanh1_layer, congress6_sigmoid1_layer, congress6_softmax1_layer])

# First MTD congress
mtd_congress1_conv1_layer = Conv2D(filters=mtd_convolution_filters[0], kernel_size=3, strides=1, padding='same')(relu2_layer)
mtd_congress1_conv2_layer = Conv2D(filters=2, name='FirstMTDCongressCoordinates',
                              kernel_size=3, strides=1, padding='same')(mtd_congress1_conv1_layer)
mtd_congress1_conv4_layer = Conv2D(filters=len(landmark_classes),
                                   name='FirstMTDCongressClass', # (count filters = count landmarks class)
                              kernel_size=3, strides=1, padding='same')(mtd_congress1_conv1_layer)
mtd_congress1_tanh1_layer = Activation('tanh', name='FirstMTDTanh')(mtd_congress1_conv2_layer)
mtd_congress1_softmax1_layer = Activation('softmax', name='FirstMTDSoftmax')(mtd_congress1_conv4_layer)
mtd_congress1_output1_layer = Concatenate(name='FirstMTDCongress')([mtd_congress1_tanh1_layer,
                                                                      mtd_congress1_softmax1_layer])

# Second MTD congress
mtd_congress2_conv1_layer = Conv2D(filters=mtd_convolution_filters[0], kernel_size=3, strides=1, padding='same')(conv5_layer)
mtd_congress2_conv2_layer = Conv2D(filters=2, name='SecondMTDCongressCoordinates',
                              kernel_size=3, strides=1, padding='same')(mtd_congress2_conv1_layer)
mtd_congress2_conv4_layer = Conv2D(filters=len(landmark_classes), name='SecondMTDCongressClass',
                              kernel_size=3, strides=1, padding='same')(mtd_congress2_conv1_layer)
mtd_congress2_tanh1_layer = Activation('tanh', name='SecondMTDTanh')(mtd_congress2_conv2_layer)
mtd_congress2_softmax1_layer = Activation('softmax', name='SecondMTDSoftmax')(mtd_congress2_conv4_layer)
mtd_congress2_output1_layer = Concatenate(name='SecondMTDCongress')([mtd_congress2_tanh1_layer,
                                                                       mtd_congress2_softmax1_layer])

# Output

output1_layer = Conv2D(filters=8, kernel_size=(3, 3), strides=1, padding='same')(dropout8_layer)
output2_layer = Conv2D(filters=10, kernel_size=(3, 3), strides=1, padding='same')(output1_layer)
output3_layer = Activation('softmax')(output2_layer)
output4_layer = Flatten()(output3_layer)


MTD = Model(input_layer,
            [congress1_output1_layer, congress2_output1_layer, congress3_output1_layer,
             congress4_output1_layer, congress5_output1_layer, congress6_output1_layer,
             mtd_congress1_output1_layer, mtd_congress2_output1_layer, output4_layer
             ]
            )

MTD.save('MTD.h5')
plot_model(MTD, show_shapes=True)
