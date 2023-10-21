import tensorflow as tf


@tf.keras.saving.register_keras_serializable()
class ConvBlock(tf.keras.layers.Layer):

    def __init__(self, num_filters, block_name='', model_name='', isTraining=True, **kwargs):
        super().__init__(name='ConvBlock')
        self.isTraining = isTraining
        self.conv1 = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same',
                                            name=model_name + '/' + block_name + '/conv_1')
        self.bn1 = tf.keras.layers.BatchNormalization(trainable=self.isTraining)
        self.relu1 = tf.keras.layers.LeakyReLU()

        self.conv2 = tf.keras.layers.Conv2D(num_filters, kernel_size=(3, 3), padding='same',
                                            name=model_name + '/' + block_name + '/conv_1')
        self.bn2 = tf.keras.layers.BatchNormalization(trainable=self.isTraining)
        self.relu2 = tf.keras.layers.LeakyReLU()

    def call(self, input_features):
        x = self.conv1(input_features)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x
