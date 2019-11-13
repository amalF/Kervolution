import tensorflow as tf
import numpy as np

class LinearKernel(tf.keras.layers.Layer):
    def __init__(self, name="LinearConv2D"):
        super(LinearKernel, self).__init__(name=name)
    def call(self, inputs):
        x, w, b = inputs
        out_channels = w.get_shape().as_list()[-1]
        w = tf.reshape(w,(-1,out_channels))
        x = tf.reshape(x,(-1,x.get_shape().as_list()[-1]))
        out = tf.matmul(x,w)
        if b is not None:
            return out + b
        return out

class DifferenceLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(DifferenceLayer, self).__init__()
    def call(self,inputs):
        x, w = inputs
        out_channels = w.get_shape().as_list()[-1]
        w = tf.reshape(w,(1,1,-1,out_channels))
        input_shape = x.get_shape().as_list()
        x = tf.reshape(x,(-1,input_shape[1]*input_shape[2],input_shape[-1]))
        x = x[:,:,:,None]
        out = tf.math.subtract(x, w)
        return out

class LPNormKernel(tf.keras.layers.Layer):
    def __init__(self,p=1):
        super(LPNormKernel, self).__init__()
        self.ord = p
        self.diff = DifferenceLayer()

    def call(self,inputs):
        x, w, b = inputs
        out = self.diff((x,w))
        out = tf.norm(out,ord=self.ord, axis=2)
        if b is not None:
            return out + b
        return out

class PolynomialKernel(tf.keras.layers.Layer):
    def __init__(self,cp=1.0, dp=3.0, trainable=False):

        super(PolynomialKernel, self).__init__(name="polynomialConv2D")
        self.dp = dp
        self.cp = cp
        self.trainable = trainable
        self.linear = LinearKernel()

    def build(self, input_shape):
        if self.trainable:
            self.cp = self.add_weight(\
                name='cp',
                shape=(),
                initializer=tf.keras.initializers.Constant(0.3),
                trainable=True,
                constraint=tf.keras.constraints.non_neg())

        super(PolynomialKernel, self).build(input_shape)

    def call(self, inputs):
        x, w, b = inputs
        conv = self.linear((x,w,None))
        s = conv + self.cp
        out =  s**self.dp
        if b is not None:
            return out +b
        return out

class SigmoidKernel(tf.keras.layers.Layer):
    def __init__(self):
        super(SigmoidKernel, self).__init__(name="sigmoidConv2D")
        self.linear = LinearKernel()

    def call(self, inputs):
        x, w, b = inputs
        out = self.linear((x,w,None))
        out = tf.math.tanh(out)
        if b is not None:
            return out +b
        return out

class GaussianKernel(tf.keras.layers.Layer):
    def __init__(self, gamma=1.0, trainable=False):
        super(GaussianKernel, self).__init__(name="gaussianConv2D")
        self.initial_gamma = gamma
        self.trainable = trainable
        self.diff = DifferenceLayer()

    def build(self, input_shape):
        if self.trainable:
            self.gamma = self.add_weight(\
                   name='gamma',
                   shape=(),
                   initializer=tf.keras.initializers.Constant(1.0),
                   trainable=True)
        else:
            self.gamma = self.initial_gamma

        super(GaussianKernel, self).build(input_shape)

    def call(self, inputs):
        x, w, b = inputs
        diff = self.diff((x,w))
        diff_norm = tf.reduce_sum(diff**2,axis=2)
        out = tf.exp(-self.gamma*diff_norm)
        if b is not None:
            return out + b
        return out

class KernelConv2D(tf.keras.layers.Conv2D):
    def __init__(self,
                filters,
                kernel_size,
                kernel_fn=GaussianKernel,
                strides=(1,1),
                padding='SAME',
                dilation_rate=(1,1),
                use_bias=False):

        super(KernelConv2D,self).__init__(filters,
                                     kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     dilation_rate=dilation_rate,
                                     use_bias=use_bias)
        self.kernel_fn = kernel_fn

    def call(self, x):
        patches = tf.image.extract_patches(\
                x,
                sizes=[1,self.kernel_size[0],self.kernel_size[1],1],
                strides=[1,self.strides[0],self.strides[1],1],
                padding=self.padding.upper(),
                rates=[1,self.dilation_rate[0],self.dilation_rate[1],1])
        output = self.kernel_fn((patches,self.kernel,self.bias))
        output_shape = [-1]+patches.get_shape().as_list()[1:3]+[output.shape[-1]]
        output = tf.reshape(output,output_shape)
        return output

def get_kernel(kernel_name, **kwargs):
    if kernel_name == 'polynomial':
        return PolynomialKernel(cp=kwargs['cp'], dp=kwargs['dp'])
    elif kernel_name == 'gaussian':
        return GaussianKernel(gamma=kwargs['gamma'])
    elif kernel_name == 'sigmoid':
        return SigmoidKernel()
    elif kernel_name == 'L1':
        return LPNormKernel(p=1)
    elif kernel_name == 'L2':
        return LPNormKernel(p=2)
    else:
        return LinearKernel()
