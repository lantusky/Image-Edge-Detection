from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, BatchNormalization, Activation, add
from keras import backend as K
from keras.layers.merge import concatenate
from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform

from my_models.ResNet import identity_block, conv_block

init = he_normal(seed=1)


def side_out(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same',
                        use_bias=False, activation=None, kernel_initializer=init)(x)
    return x


def u_res_side_fuse(input_shape=None):

    inputs = Input(shape=input_shape)       # 320, 480, 3
    # Normalization
    x = Lambda(lambda x: x / 255, name='pre-process')(inputs)

    x = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1')(x)
    x = BatchNormalization(axis=-1, name='bn_conv1')(x)
    x = Activation('relu', name='act1')(x)  # 320, 480, 3

    # Block 1
    c1 = conv_block(x, 3, (8, 8, 32), stage=1, block='a', strides=(1, 1))
    c1 = identity_block(c1, 3, (8, 8, 32), stage=1, block='b')      # 320, 480, 3

    # Block 2
    c2 = conv_block(c1, 3, (16, 16, 64), stage=2, block='a', strides=(2, 2))
    c2 = identity_block(c2, 3, (16, 16, 64), stage=2, block='b')    # 160, 240, 3

    # Block 3
    c3 = conv_block(c2, 3, (32, 32, 128), stage=3, block='a', strides=(2, 2))
    c3 = identity_block(c3, 3, (32, 32, 128), stage=3, block='b')   # 80, 120, 3

    # Block 4
    c4 = conv_block(c3, 3, (64, 64, 256), stage=4, block='a', strides=(2, 2))
    c4 = identity_block(c4, 3, (64, 64, 256), stage=4, block='b')   # 40, 60, 3

    # Block 5
    c5 = conv_block(c4, 3, (128, 128, 512), stage=5, block='a', strides=(2, 2))
    c5 = identity_block(c5, 3, (128, 128, 512), stage=5, block='b') # 20, 30, 3
    s1 = side_out(c5, 16)

    # Block 6
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')        # 40, 60, 3
    c6 = conv_block(u6, 3, (64, 64, 256), stage=6, block='a', strides=(1, 1))
    c6 = identity_block(c6, 3, (64, 64, 256), stage=6, block='b')
    s2 = side_out(c6, 8)

    # Block 7
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')        # 80, 120, 3
    c7 = conv_block(u7, 3, (32, 32, 128), stage=7, block='a', strides=(1, 1))
    c7 = identity_block(c7, 3, (32, 32, 128), stage=7, block='b')
    s3 = side_out(c7, 4)

    # Block 8
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')        # 160, 240, 3
    c8 = conv_block(u8, 3, (16, 16, 64), stage=8, block='a', strides=(1, 1))
    c8 = identity_block(c8, 3, (16, 16, 64), stage=8, block='b')
    s4 = side_out(c8, 2)

    # Block 9
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')        # 320, 480, 3
    c9 = conv_block(u9, 3, (8, 8, 32), stage=9, block='a', strides=(1, 1))
    c9 = identity_block(c9, 3, (8, 8, 32), stage=9, block='b')
    s5 = side_out(c9, 1)

    # fuse
    fuse = concatenate(inputs=[s1, s2, s3, s4, s5], axis=-1)
    fuse = Conv2D(1, (1, 1), padding='same', activation=None)(fuse)       # 320 480 1

    # outputs
    o1    = Activation('sigmoid', name='o1')(s1)
    o2    = Activation('sigmoid', name='o2')(s2)
    o3    = Activation('sigmoid', name='o3')(s3)
    o4    = Activation('sigmoid', name='o4')(s4)
    o5    = Activation('sigmoid', name='o5')(s5)
    ofuse = Activation('sigmoid', name='ofuse')(fuse)

    model = Model(inputs=[inputs], outputs=[o1, o2, o3, o4, o5, ofuse])

    return model