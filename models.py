import tensorflow as tf
from layers import *

class LeNet5():                                                               
    def __init__(self,                                                        
                 num_classes=10,                                              
                 keep_prob=1.0):
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


class LeNet5KNN(LeNet5):
    def __init__(self,
                 num_classes=10,
                 kernel_fn=get_kernel("linear"),
                 pooling="max",
                 keep_prob=1.0):
        super(LeNet5KNN, self).__init__(num_classes=num_classes, keep_prob=keep_prob)

        self.conv1 = KernelConv2D(10,
                                  kernel_size=(3,3),
                                  kernel_fn=kernel_fn)

        self.conv2 = KernelConv2D(20,
                                  kernel_size=(3,3),
                                  kernel_fn=kernel_fn) 
        if pooling == 'avg':
            self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=2,
                                   strides=2,
                                    padding="VALID")

            self.pool2 = tf.keras.layers.AveragePooling2D(pool_size=2,
                                strides=2,
                                padding="VALID")


def get_model(model_name, input_shape, num_classes=10, keep_prob=1.0, **kwargs):
    if model_name=="KNN":
        archi = LeNet5KNN(num_classes=num_classes,
                        keep_prob=keep_prob,
                        **kwargs)
    elif model_name=='CNN':
        archi = LeNet5(num_classes=num_classes,
                         keep_prob=keep_prob)
    else:
        raise ValueError("Unknown model name {}".format(model_name)) 

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = archi.conv1(inputs)
    x = tf.keras.layers.ReLU()(x)
    x = archi.pool1(x)
    x = archi.conv2(x)
    x = tf.keras.layers.ReLU()(x)
    x = archi.pool2(x)
    x = tf.keras.layers.Flatten()(x)
    x = archi.fc3(x)
    if keep_prob<1.0:
        x = archi.drop1(x)
    logits = archi.fc4(x)

    return tf.keras.Model(inputs=inputs, outputs=logits,\
            name="LeNet5-{}".format(model_name))
