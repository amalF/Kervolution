import tensorflow as tf
import numpy as np

class LinearKernel(tf.keras.layers.Layer):
    def __init__(self):
        super(LinearKernel, self).__init__()
    def call(self, x, w, b):
        out_channels = w.get_shape().as_list()[-1]
        w = tf.reshape(w,(-1,out_channels))
        x = tf.reshape(x,(-1,x.get_shape().as_list()[-1]))
        out = tf.matmul(x,w)
        if b is not None:
            return out + b
        return out

class LPNormKernel(tf.keras.layers.Layer):
    def __init__(self,p=1):
        super(LPNormKernel, self).__init__()
        self.ord = p

    def call(self,x,w,b):
        out_channels = w.get_shape().as_list()[-1]
        w = tf.reshape(w,(1,1,-1,out_channels))
        input_shape = x.get_shape().as_list()
        x = tf.reshape(x,(-1,input_shape[1]*input_shape[2],input_shape[-1]))
        out = x[:,:,:,None] - w
        out = tf.norm(out,ord=self.ord, axis=2)
        if b is not None:
            return out + b
        return out

class PolynomialKernel(LinearKernel):
    def __init__(self,cp=1.0, dp=3, train_cp=True):
        super(PolynomialKernel, self).__init__()
        self.initial_cp = cp
        self.dp = dp
        self.train_cp = train_cp
    def build(self, input_shape):
        self.cp = self.add_variable(\
                name='cp',
                shape=(1,),
                initializer=tf.keras.initializers.constant(self.initial_cp),
                trainable=self.train_cp)
        self.built = True
    def call(self, x, w, b):
        conv = super(PolynomialKernel,self).call(x,w,b)
        s = conv + self.cp
        return tf.pow(s,self.dp)

class GaussianKernel(LPNormKernel):
    def __init__(self, gamma=1, train_gamma=True):
        super(GaussianKernel, self).__init__(p=2)
        self.initial_gamma = gamma
        self.train_gamma = train_gamma

    def build(self, input_shape):
        self.gamma = self.add_variable(\
                   name='gamma',
                   shape=(1,),
                   initializer=tf.keras.initializers.constant(self.initial_gamma),
                   trainable=self.train_gamma)
        self.built = True

    def call(self, x, w, b):
        diff_norm = super(GaussianKernel,self).call(x,w,None)
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
                padding='VALID',
                dilation_rate=(1,1),
                use_bias=False):

        super(KernelConv2D,self).__init__(filters,
                                     kernel_size,
                                     strides=strides,
                                     padding=padding,
                                     dilation_rate=dilation_rate,
                                     use_bias=use_bias)
        self.kernel_fn = kernel_fn()

    def call(self, x):
        patches = tf.image.extract_image_patches(\
                x,
                sizes=[1,self.kernel_size[0],self.kernel_size[1],1],
                strides=[1,self.strides[0],self.strides[1],1],
                padding=self.padding.upper(),
                rates=[1,self.dilation_rate[0],self.dilation_rate[1],1])
        output = self.kernel_fn(patches,self.kernel,self.bias)
        output_shape = [-1]+patches.get_shape().as_list()[1:3]+[output.shape[-1]]
        output = tf.reshape(output,output_shape)
        return output
