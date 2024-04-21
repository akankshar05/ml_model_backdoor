# import tensorflow as tf
# from copy import deepcopy


# class DataSet(object):
#     def __init__(self, load_fun, target, classes, augmentation=True):
#         (self.x_train, self.y_train), (self.x_test, self.y_test) = load_fun
#         # self.x_train=self.x_train[:4]
#         # self.y_train=self.y_train[:4]
#         # self.x_test=self.x_test[:4]
#         # self.y_test=self.y_test[:4]
#         self.x_train = self.x_train.astype("float32") / 255.
#         self.x_test = self.x_test.astype("float32") / 255.

#         self.train_samples = self.x_train.shape[0]
#         self.target = target
#         self.classes = classes
#         self.augmentation = augmentation
        
#         self.x_train_poison, self.y_train_poison, self.x_test_poison, self.y_test_poison, =\
#             deepcopy(self.x_train), deepcopy(self.y_train), deepcopy(self.x_test), deepcopy(self.y_test)

#         self.image_gen = self.preprocess()
#         self.image_gen_poison = self.preprocess_poison()

#     def preprocess(self):
#         if self.augmentation:
#             image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rotation_range=20,
#                 width_shift_range=0.2,
#                 height_shift_range=0.2,
#                 horizontal_flip=True,
#             )
#         else:
#             image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator()
#         image_gen_train.fit(self.x_train)
#         return image_gen_train

#     def preprocess_poison(self):
#         image_size = self.x_train_poison.shape[1]
#         pattern_a = int(image_size * 0.75)
#         pattern_b = int(image_size * 0.9375)

#         for i in range(len(self.x_train_poison)):
#             self.x_train_poison[i, pattern_a:pattern_b, pattern_a:pattern_b] = 1
#             self.y_train_poison[i] = self.target
#         for i in range(len(self.x_test_poison)):
#             self.x_test_poison[i, pattern_a:pattern_b, pattern_a:pattern_b] = 1
#             self.y_test_poison[i] = self.target

#         if self.augmentation:
#             image_gen_poison = tf.keras.preprocessing.image.ImageDataGenerator()
#         else:
#             image_gen_poison = tf.keras.preprocessing.image.ImageDataGenerator()
#         image_gen_poison.fit(self.x_train_poison)
#         return image_gen_poison

#     def ds_data(self, batch_size):
#         ds_train = self.image_gen.flow(
#             self.x_train, self.y_train, batch_size=batch_size
#         )
#         ds_train_backdoor = self.image_gen_poison.flow(
#                self.x_train_poison, self.y_train, batch_size=batch_size
#         )
#         ds_train_backdoor_y_poison = self.image_gen_poison.flow(
#                self.x_train_poison, self.y_train_poison, batch_size=batch_size
#            )
        
#         ds_test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)) \
#             .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
#         ds_test_backdoor = tf.data.Dataset.from_tensor_slices((self.x_test_poison, self.y_test)) \
#             .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
#         ds_test_backdoor_y_poison = tf.data.Dataset.from_tensor_slices((self.x_test_poison, self.y_test_poison)) \
#             .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
            
#         return ds_train, ds_train_backdoor, ds_train_backdoor_y_poison, ds_test, ds_test_backdoor, ds_test_backdoor_y_poison

# def Cifar10(target=0):
#     cifar10 = DataSet(load_fun=tf.keras.datasets.cifar10.load_data(), target=target, classes=10)
#     return cifar10


import tensorflow as tf
from copy import deepcopy
import numpy as np
# from tensorflow.python.ops.numpy_ops import np_config
# np_config.enable_numpy_behavior()
import matplotlib.pyplot as plt

class DataSet(object):
    def __init__(self, load_fun, target, classes, augmentation=True):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = load_fun
        
        # self.x_train=self.x_train[:2]
        # self.y_train= self.y_train[:2]
        # self.x_test=self.x_test[:2]
        # self.y_test=self.y_test[:2]
        
        def show_image(self, image):
            plt.imshow(image)
            plt.axis('off')
            plt.show()
            
        # print(self.x_train[0])
        # print(self.x_train[0].shape)
            
        # show_image(self,self.x_train[0])
            
        #resize the numpy arrays for VGG16 - the CV alg requires a 32 x 32 array rather that 28 x 28.
   
        self.x_train = tf.pad(self.x_train, [[0, 0], [2,2], [2,2]])/255
        self.x_test = tf.pad(self.x_test, [[0, 0], [2,2], [2,2]])/255
        self.x_train = tf.expand_dims(self.x_train, axis=3, name=None)
        self.x_test = tf.expand_dims(self.x_test, axis=3, name=None)
        self.x_train = tf.repeat(self.x_train, 3, axis=3)
        self.x_test = tf.repeat(self.x_test, 3, axis=3)
        self.x_train.shape

  
        # self.x_train = self.x_train.reshape(self.x_train.shape[0],self.x_train.shape[1],self.x_train.shape[2],1)

        # self.x_train = self.x_train.astype("float32") / 255.
        # self.x_test = self.x_test.astype("float32") / 255.

        self.train_samples = self.x_train.shape[0]
        self.target = target
        self.classes = classes
        self.augmentation = augmentation

        self.x_train_poison, self.y_train_poison, self.x_test_poison, self.y_test_poison, =\
            deepcopy(self.x_train), deepcopy(self.y_train), deepcopy(self.x_test), deepcopy(self.y_test)

        self.image_gen = self.preprocess()
        self.image_gen_poison = self.preprocess_poison()

    def preprocess(self):
        if self.augmentation:
            image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
            )
        else:
            image_gen_train = tf.keras.preprocessing.image.ImageDataGenerator()
        image_gen_train.fit(self.x_train)
        return image_gen_train

    def preprocess_poison(self):
        image_size = self.x_train_poison.shape[1]
        pattern_a = int(image_size * 0.75)
        pattern_b = int(image_size * 0.9375)
        self.x_train_poison=self.x_train_poison.numpy()
        self.x_test_poison=self.x_test_poison.numpy()

        for i in range(len(self.x_train_poison)):
            self.x_train_poison[i, pattern_a:pattern_b, pattern_a:pattern_b] = 1
            self.y_train_poison[i] = self.target
        for i in range(len(self.x_test_poison)):
            self.x_test_poison[i, pattern_a:pattern_b, pattern_a:pattern_b] = 1
            self.y_test_poison[i] = self.target

        #convert back to tensor
        self.x_train_poison=tf.convert_to_tensor(self.x_train_poison, dtype=tf.float32)
        self.x_test_poison=tf.convert_to_tensor(self.x_test_poison, dtype=tf.float32)


        if self.augmentation:
            image_gen_poison = tf.keras.preprocessing.image.ImageDataGenerator()
        else:
            image_gen_poison = tf.keras.preprocessing.image.ImageDataGenerator()
        image_gen_poison.fit(self.x_train_poison)
        return image_gen_poison

    def ds_data(self, batch_size, backdoor=True):
        ds_train = self.image_gen.flow(
            self.x_train, self.y_train, batch_size=batch_size
        )

        if backdoor:
            ds_train_backdoor = self.image_gen_poison.flow(
                self.x_train_poison, self.y_train_poison, batch_size=batch_size
            )
        else:
            ds_train_backdoor = self.image_gen_poison.flow(
                self.x_train_poison, self.y_train, batch_size=batch_size
            )

        ds_test = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        ds_test_backdoor = tf.data.Dataset.from_tensor_slices((self.x_test_poison, self.y_test_poison)) \
            .batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        exclude_target_index = tf.where(tf.squeeze(self.y_test) != 0)
        ds_test_backdoor_exclude_target = tf.data.Dataset.from_tensor_slices(
            (tf.gather_nd(self.x_test_poison, exclude_target_index),
             tf.gather_nd(self.y_test_poison, exclude_target_index))
        ).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train, ds_train_backdoor, ds_test, ds_test_backdoor, ds_test_backdoor_exclude_target


def MNIST(target=0):
    MNIST = DataSet(load_fun=tf.keras.datasets.mnist.load_data(), target=target, classes=10)
    return MNIST