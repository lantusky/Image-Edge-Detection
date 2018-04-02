from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Input, Lambda, MaxPooling2D, Activation
from keras.layers.merge import concatenate
from keras.initializers import glorot_normal, glorot_uniform, he_normal, he_uniform

init = he_normal(seed=1)


# Build U-Net model
def u_net_ori(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    # Block 1
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1a', kernel_initializer=init)(s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_1b', kernel_initializer=init)(c1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    # Block 2
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2a', kernel_initializer=init)(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_2b', kernel_initializer=init)(c2)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    # Block 3
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3a', kernel_initializer=init)(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_3b', kernel_initializer=init)(c3)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    # Block 4
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_4a', kernel_initializer=init)(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_4b', kernel_initializer=init)(c4)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)
    # Block 5
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_5a', kernel_initializer=init)(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_5b', kernel_initializer=init)(c5)
    # Block 6
    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_6a', kernel_initializer=init)(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_6b', kernel_initializer=init)(c6)
    # Block 7
    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_7a', kernel_initializer=init)(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_7b', kernel_initializer=init)(c7)
    # Block 8
    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_8a', kernel_initializer=init)(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_8b', kernel_initializer=init)(c8)
    # Block 9
    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_9a', kernel_initializer=init)(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_9b', kernel_initializer=init)(c9)
    # Output
    outputs = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer=init)(c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return model


def side_out(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)

    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same',
                        use_bias=False, activation=None, kernel_initializer=init)(x)
    return x


# Build U-Net model
def u_net_fuse(input_shape=None):

    inputs = Input(shape=input_shape)
    # Normalization
    s = Lambda(lambda x: x / 255)(inputs)
    # Block 1
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1a', kernel_initializer=init)(s)
    c1 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1b', kernel_initializer=init)(c1)
    p1 = MaxPooling2D((2, 2), name='pool_1')(c1)
    # Block 2
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2a', kernel_initializer=init)(p1)
    c2 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2b', kernel_initializer=init)(c2)
    p2 = MaxPooling2D((2, 2), name='pool_2')(c2)
    # Block 3
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3a', kernel_initializer=init)(p2)
    c3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_3b', kernel_initializer=init)(c3)
    p3 = MaxPooling2D((2, 2), name='pool_3')(c3)
    # Block 4
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4a', kernel_initializer=init)(p3)
    c4 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4b', kernel_initializer=init)(c4)
    p4 = MaxPooling2D((2, 2), name='pool_4')(c4)

    # Block 5
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_5a', kernel_initializer=init)(p4)
    c5 = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_5b', kernel_initializer=init)(c5)
    s1 = side_out(c5, 16)

    # Block 6
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', name='upconv_6', kernel_initializer=init)(c5)
    u6 = concatenate([u6, c4], name='concat_6')
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_6a', kernel_initializer=init)(u6)
    c6 = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_6b', kernel_initializer=init)(c6)
    s2 = side_out(c6, 8)

    # Block 7
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', name='upconv_7', kernel_initializer=init)(c6)
    u7 = concatenate([u7, c3], name='concat_7')
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_7a', kernel_initializer=init)(u7)
    c7 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_7b', kernel_initializer=init)(c7)
    s3 = side_out(c7, 4)

    # Block 8
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', name='upconv_8', kernel_initializer=init)(c7)
    u8 = concatenate([u8, c2], name='concat_8')
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_8a', kernel_initializer=init)(u8)
    c8 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_8b', kernel_initializer=init)(c8)
    s4 = side_out(c8, 2)

    # Block 9
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', name='upconv_9', kernel_initializer=init)(c8)
    u9 = concatenate([u9, c1], name='concat_9')
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_9a', kernel_initializer=init)(u9)
    c9 = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_9b', kernel_initializer=init)(c9)
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