import tensorflow as tf
from models import resnet
from layers import *

class ResNetKNN(resnet.ResNet):
    def __init__(self, num_blocks, kernel_fn, num_classes=10):
        super(ResNetKNN, self).__init__(num_blocks, num_classes=num_classes)

        self.conv1 = KernelConv2D(16, kernel_size=(3,3),
                kernel_fn=kernel_fn)

def ResNet18(kernel_fn=get_kernel("linear"), num_classes=10):
    return ResNetKNN([2,2,2,2], kernel_fn, num_classes= num_classes)
