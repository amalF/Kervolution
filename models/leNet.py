import tensorflow as tf

class LeNet5(tf.keras.Model):                                                               
    def __init__(self,                                                        
                 num_classes=10,                                              
                 keep_prob=1.0):
        super(LeNet5, self).__init__()

        self.keep_prob = keep_prob
        self.conv1 = tf.keras.layers.Conv2D(10,
                                           (3,3),
                                           padding='same')
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=2,
                                               strides=2,
                                               padding="VALID")
        self.conv2 = tf.keras.layers.Conv2D(20,
                                            (3,3),
                                            padding='same')
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=2,
                                                strides=2,
                                                padding="VALID")
        self.fc3 = tf.keras.layers.Dense(320,
                                         activation='relu')
        if self.keep_prob<1.0:
            self.drop1 = tf.keras.layers.Dropout(1-keep_prob)
        self.fc4 = tf.keras.layers.Dense(num_classes,
                                        activation="softmax")


    def call(self, inputs):
        x = self.conv1(inputs)
        x = tf.keras.layers.ReLU()(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = tf.keras.layers.ReLU()(x)
        x = self.pool2(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc3(x)
        if self.keep_prob<1.0:
            x = self.drop1(x)
        logits = self.fc4(x)
        return logits
