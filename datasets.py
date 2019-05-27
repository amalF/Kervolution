import numpy as np
import os
import multiprocessing
import tensorflow as tf

class DataSet(object):
    """
    Create a data set.
    """

    def __init__(self,
                 data,
                 image_dims,
                 use_distortion,
                 shuffle,
                 repeat,
                 nThreads):

        self.image_height = image_dims[0]
        self.image_width = image_dims[1]
        self.image_depth = image_dims[2]
        self.use_distortion = use_distortion
        self.data = data
        self.shuffle = shuffle
        self.repeat = repeat
        if nThreads:
            self.nrof_threads = nThreads
        else:
            self.nrof_threads = multiprocessing.cpu_count()

    def _map_fn(self, data_example):
        pass

    def make_batch(self,
                   batch_size,
                   map_fn = None,
                   filter_fn = None):
        """
        Make data batches
        """
        # Extract data
        # This can be costy if the dataset if too large
        # TODO consider using Iterator
        dataset = tf.data.Dataset.from_tensor_slices(self.data)

        #Shuffle the data before
        if self.shuffle:
            dataset = dataset.shuffle(128*batch_size)

        # Transform data

        if filter_fn:
            dataset = dataset.filter(filter_fn)

        dataset = dataset.map(self._map_fn,
                              num_parallel_calls=self.nrof_threads)

        # Batch it up.
        dataset = dataset.batch(batch_size)
        # Repeat
        dataset = dataset.repeat(self.repeat).prefetch(batch_size)

        return dataset

class Cifar10DataSet(DataSet):
    def __init__(self,
                image_dims = (32,32,3),
                subset='train', 
                use_distortion=True,
                shuffle=False,
                repeat=1,
                nThreads=None):

        train_data, test_data = tf.keras.datasets.cifar10.load_data()
        indexes = np.arange(len(train_data[0]))
        #np.random.shuffle(indexes)

        if subset == "train":
            train_indexes = indexes[:45000]
            train_images = train_data[0][train_indexes,:,:,:]
            train_labels = train_data[1][train_indexes,:]
            data = (train_images, train_labels)
            self.num_samples = len(train_images)
        elif subset == "valid":
            val_indexes = indexes[45000:]
            val_images = train_data[0][val_indexes,:,:,:]
            val_labels = train_data[1][val_indexes,:]
            data = (val_images, val_labels)
            self.num_samples = len(val_images)
        elif subset == "test":
            data = test_data

        self.subset = subset

        super(Cifar10DataSet, self).__init__(data,
                                             image_dims,
                                             use_distortion,
                                             shuffle,
                                             repeat,
                                             nThreads)
    @tf.function
    def _map_fn(self, image, label):
        """
        Apply transformations on the data
        """
        image, label = self.preprocess(image, label)
        return image, label

    def preprocess(self, image, label):
        """Preprocess a single image in [height, width, depth] layout."""
        image = tf.cast(image, tf.float32)/255.0
        #image = tf.clip_by_value(image, 0, 255.0)
        #image = image / 127.5 -1
        label = tf.cast(label, tf.int64)

        #if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch 
            #image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            #image = tf.image.random_crop(image, [self.image_height, self.image_width, self.image_depth])
            #image = tf.image.random_flip_left_right(image)

        #image = image/255.0 #tf.image.per_image_standardization(image)
        return image, label


class MnistDataSet(DataSet):
    def __init__(self,
                image_dims = (28,28,1),
                subset='train',
                use_distortion=True,
                shuffle=False,
                repeat=1,
                nThreads=None):
        #Load Data
        train_data, test_data = tf.keras.datasets.mnist.load_data()
        self.subset = subset
        if self.subset == "train":
            data = train_data
            self.num_samples = len(train_data[0])
        else:
            data = test_data
            self.num_samples = len(test_data[0])

        super(MnistDataSet, self).__init__(data,
                                           image_dims,
                                           use_distortion,
                                           shuffle,
                                           repeat,
                                           nThreads)
    @tf.function
    def _map_fn(self, image, label):
        return self.preprocess(image, label)

    def preprocess(self, image, label):
        image = tf.cast(image, tf.float32)/255.0
        label = tf.cast(label, tf.int64)
        return image, label
