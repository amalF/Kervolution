import tensorflow as tf
from models import leNet
from layers import get_kernel, KernelConv2D

class LeNet5KNN(leNet.LeNet5):
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
