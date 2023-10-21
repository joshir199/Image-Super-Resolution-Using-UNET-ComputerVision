import tensorflow as tf

@tf.keras.saving.register_keras_serializable()
class DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, skip_layers, block_name='', model_name='', **kwargs):
        super().__init__(name='DecoderBlock')
        self.skip_layer = skip_layers
        self.convTranspose = tf.keras.layers.Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2),
                                            name=self.model_name + '/' + block_name + '/convTranspose_1')
        self.concatenate = tf.keras.layers.Concatenate()
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, input_features):
        x = self.convTranspose(input_features)
        x = self.concatenate([x, self.skip_layer])
        x = self.dropout(x)
        return x