import tensorflow as tf


class ImageResolutionModel:

    def __init__(self, kernel_size, SIZE, channels, isTraining=True):
        self.kernel_size = kernel_size
        self.SIZE = SIZE
        self.channels = channels
        self.isTraining = isTraining

    def conv_block(self, input_features, num_filters):
        x = tf.keras.layers.Conv2D(num_filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same')(
            input_features)
        x = tf.keras.layers.BatchNormalization(trainable=self.isTraining)(x)
        x = tf.keras.layers.LeakyReLU()(x)

        x = tf.keras.layers.Conv2D(num_filters, kernel_size=(self.kernel_size, self.kernel_size), padding='same')(x)
        x = tf.keras.layers.BatchNormalization(trainable=self.isTraining)(x)
        x = tf.keras.layers.LeakyReLU()(x)
        return x

    def encoder_block(self, input_features, num_filters):
        skip = self.conv_block(input_features, num_filters)
        x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(skip)
        return skip, x

    def decoder_block(self, input_features, num_filters, skip_layer):
        x = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2))(input_features)
        x = tf.keras.layers.Concatenate()([x, skip_layer])
        x = tf.keras.layers.Dropout(0.3)(x)
        return x

    def unet_model(self):

        inputs = tf.keras.layers.Input((self.SIZE, self.SIZE, self.channels), name='unet_model')

        x = tf.keras.layers.Conv2D(filters=self.channels, kernel_size=(1, 1), padding='same')(inputs)

        skip_1, x = self.encoder_block(x, 32)
        skip_2, x = self.encoder_block(x, 64)
        skip_3, x = self.encoder_block(x, 128)
        skip_4, x = self.encoder_block(x, 256)
        skip_5, x = self.encoder_block(x, 512)

        bottom = self.conv_block(x, 1024)

        d1 = self.decoder_block(bottom, 512, skip_5)
        d2 = self.decoder_block(d1, 256, skip_4)
        d3 = self.decoder_block(d2, 128, skip_3)
        d4 = self.decoder_block(d3, 64, skip_2)
        d5 = self.decoder_block(d4, 32, skip_1)

        outputs = tf.keras.layers.Conv2D(self.channels, kernel_size=(1, 1), padding='same')(d5)

        model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

        return model
