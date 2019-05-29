import tensorflow as tf

class LeNet5():#tf.keras.Model):
    def __init__(self,
                 num_classes=10,
                 convLayer=tf.keras.layers.Conv2D,
                 kernel_fn=None,
                 keep_prob=0.5):
        #super(LeNet5, self).__init__()
        self.conv1 = convLayer(16,
                               kernel_size=(5,5),
                               kernel_fn=kernel_fn,
                               use_bias=False)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2,
                                               strides=2,
                                               padding="VALID")
        self.conv2 = convLayer(32,
                               kernel_size=(5,5),
                               kernel_fn=kernel_fn,
                               use_bias=False)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2,
                                               strides=2,
                                               padding="VALID")
        self.fc3 = tf.keras.layers.Dense(1024,
                                         activation='relu')
        #self.drop1 = tf.keras.layers.Dropout(1-keep_prob)
        self.fc4 = tf.keras.layers.Dense(num_classes,
                                        activation="softmax",
                                        use_bias=True)
    def build(self, input_shape):
        super(LeNet5, self).build(input_shape)
        self.build=True

    def call(self, input_shape):
        end_points = {}
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc3(x)
        logits = self.fc4(x)
        return tf.keras.Model(inputs=inputs, outputs=logits, name ="LeNet5")



