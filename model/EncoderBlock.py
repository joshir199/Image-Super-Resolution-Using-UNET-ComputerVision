import tensorflow as tf

from model.ConvBlock import ConvBlock

@tf.keras.saving.register_keras_serializable()
class EncoderBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, block_name='', model_name='', **kwargs):
        super().__init__(name='EncoderBlock')
        self.conv = ConvBlock(num_filters, block_name, model_name)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, input_features):
        skip_layer = self.conv(input_features)
        x = self.pool1(skip_layer)
        return skip_layer, x
