import tensorflow as tf


class ImageSuperResolutionUnetModel(tf.keras.Model):

    def __init__(self, SIZE, channels, isTraining=True):
        """
        :param SIZE: Image size
        :param channels: number of channels in input images
        :param isTraining: is training or not
        """
        self.model_name = 'UNet Model'
        self.isTraining = isTraining
        self.SIZE = SIZE
        self.channels = channels
        self.input = tf.keras.layers.Input(shape=(self.SIZE, self.SIZE, self.channels))
        self.conv2d_1 = tf.keras.layers.Conv2D(self.channels, kernel_size=(1, 1), padding='same',
                                               name='UNet Model/Intput/Conv_1')
        self.output = tf.keras.layers.Conv2D(self.channels, kernel_size=(1, 1), padding='same',
                                             name='UNet model/output/Conv_out')

    def call(self):
        """
        :return: returns UNET model
        """

        input = self.input
        x = self.conv2d_1(input)

        skip_1, x = self.encoder_block(x, 32, 'encode_1', self.model_name)
        skip_2, x = self.encoder_block(x, 64, 'encode_2', self.model_name)
        skip_3, x = self.encoder_block(x, 128, 'encode_3', self.model_name)
        skip_4, x = self.encoder_block(x, 256, 'encode_4', self.model_name)
        skip_5, x = self.encoder_block(x, 512, 'encode_5', self.model_name)

        bottom = self.conv_block(x, 1024, 'bottom', self.model_name)

        d1 = self.decoder_block(bottom, 512, skip_5, 'decode_1')
        d2 = self.decoder_block(d1, 256, skip_4, 'decode_2')
        d3 = self.decoder_block(d2, 128, skip_3, 'decode_3')
        d4 = self.decoder_block(d3, 64, skip_2, 'decode_4')
        d5 = self.decoder_block(d4, 32, skip_1, 'decode_5')

        outputs = self.output(d5)

        model = tf.keras.models.Model(inputs=[input], outputs=[outputs])

        return model
