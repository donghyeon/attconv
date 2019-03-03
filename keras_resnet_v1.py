# Implementation of resnet_v1 with keras layers.

import tensorflow as tf


class ResnetV1(tf.keras.Model):
    def __init__(self, num_classes=None, global_pool=True, spatial_squeeze=True, dtype=tf.float32, *args, **kwargs):
        super(ResnetV1, self).__init__(*args, **kwargs)
        self._num_classes = num_classes
        self._global_pool = global_pool
        self._spatial_squeeze = spatial_squeeze
        self._dtype = dtype

        self._conv1 = self.conv1()
        self._block1 = self.block1()
        self._block2 = self.block2()
        self._block3 = self.block3()
        self._block4 = self.block4()
        if num_classes:
            self._dense = self.dense()

    def conv1(self):
        return BasicConvUnit(64, 7, strides=2, padding='valid', name='conv1')

    def block1(self):
        pass

    def block2(self):
        pass

    def block3(self):
        pass

    def block4(self):
        pass

    def dense(self):
        return tf.keras.layers.Dense(self._num_classes, name='logits')

    def stack_blocks(self):
        blocks = [self._conv1,
                  self._block1,
                  self._block2,
                  self._block3,
                  self._block4]
        return blocks

    def call(self, inputs, training=False):
        x = inputs
        for block in self.stack_blocks():
            x = block(x, training=training)
        if self._global_pool:
            x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        if self._num_classes:
            x = self._dense(x)
        if self._spatial_squeeze:
            x = tf.squeeze(x, axis=[1, 2])
        return x


class ResnetBlock(tf.keras.Model):
    def __init__(self, depth, num_units, dtype=tf.float32, *args, **kwargs):
        super(ResnetBlock, self).__init__(*args, **kwargs)
        self._depth = depth
        self._num_units = num_units
        self._dtype = dtype
        self._units = self.stack_units()

    def stack_units(self):
        units = []
        for i in range(self._num_units):
            bottleneck_class = Stride2BottleneckV1 if i == 0 else BottleneckV1
            unit = bottleneck_class(self._depth, dtype=self._dtype, name='unit_{}/bottleneck_v1'.format(i + 1))
            units.append(unit)
        return units

    def call(self, inputs, training=False):
        x = inputs
        for unit in self._units:
            x = unit(x, training=training)
        return x


class BottleneckV1(tf.keras.Model):
    def __init__(self, depth, activation='relu', dtype=tf.float32, name='bottleneck_v1', *args, **kwargs):
        super(BottleneckV1, self).__init__(name=name, *args, **kwargs)
        self._depth = depth
        self._dtype = dtype

        self._conv1 = self.conv1()
        self._conv2 = self.conv2()
        self._conv3 = self.conv3()
        self._shortcut = lambda x, training: x
        self._act = tf.keras.layers.Activation(activation)

    def conv1(self):
        return BasicConvUnit(filters=self._depth, kernel_size=1, dtype=self._dtype, name='conv1')

    def conv2(self):
        return BasicConvUnit(filters=self._depth, kernel_size=3, dtype=self._dtype, name='conv2')

    def conv3(self):
        return BasicConvUnit(filters=4 * self._depth, kernel_size=1, activation=None, dtype=self._dtype, name='conv3')

    def shortcut(self):
        return BasicConvUnit(filters=4 * self._depth, kernel_size=1, dtype=self._dtype, name='shortcut')

    def build(self, input_shape):
        if input_shape[-1] != 4 * self._depth:
            self._shortcut = self.shortcut()
        self.built = True

    def call(self, inputs, training=False):
        x = inputs
        x = self._conv1(x, training=training)
        x = self._conv2(x, training=training)
        x = self._conv3(x, training=training)
        x = self._act(x + self._shortcut(inputs, training=training))
        return x


class Stride2BottleneckV1(BottleneckV1):
    def conv1(self):
        return BasicConvUnit(filters=self._depth, kernel_size=1, strides=2, padding='valid', dtype=self._dtype,
                             name='conv1')

    def shortcut(self):
        return BasicConvUnit(filters=4 * self._depth, kernel_size=1, strides=2, padding='valid', dtype=self._dtype,
                             name='shortcut')

    @staticmethod
    def pad_inputs_for_valid_conv(inputs):
        inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]])
        # This means that pad inputs [[batch_beg, batch_end], [height_beg, ...], ..., [..., depth_end]] times
        # In case height or width is even, default Conv2D with strides=2 and padding='same' pads the inputs by
        # tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]) resulting into different outputs.
        return inputs

    def call(self, inputs, training=False):
        x = self.pad_inputs_for_valid_conv(inputs)
        x = super(Stride2BottleneckV1, self).call(x, training=training)
        return x


# Basic convolutional block. Conv2D + BatchNormalization + (ReLU)
class BasicConvUnit(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, padding='same', activation='relu', dtype=tf.float32,
                 *args, **kwargs):
        super(BasicConvUnit, self).__init__(*args, **kwargs)
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False, dtype=dtype,
                                           name='conv2d')
        self.bn = tf.keras.layers.BatchNormalization(dtype=dtype, name='BatchNorm')  # TODO: freeze batch norm params
        self._activation = activation
        if activation:
            self.act = tf.keras.layers.Activation(activation, dtype=dtype)

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self._activation:
            x = self.act(x)
        return x


class ResnetV1_50(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 4, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 6, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, dtype=self._dtype, name='block4')


class ResnetV1_101(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 4, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 23, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, dtype=self._dtype, name='block4')


class ResnetV1_152(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 8, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 36, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, dtype=self._dtype, name='block4')


class ResnetV1_200(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 24, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 36, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, dtype=self._dtype, name='block4')
