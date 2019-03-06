# Implementation of resnet_v1 with keras layers.

import os
from collections import OrderedDict
import tensorflow as tf


class ResnetV1(tf.keras.Model):
    def __init__(self, num_classes=None, global_pool=True, spatial_squeeze=True, dtype=tf.float32, *args, **kwargs):
        super(ResnetV1, self).__init__(*args, **kwargs)
        self._num_classes = num_classes
        self._global_pool = global_pool
        self._spatial_squeeze = spatial_squeeze
        self._dtype = dtype

        self._conv1 = self.conv1()
        self._pool1 = self.pool1()
        self._block1 = self.block1()
        self._block2 = self.block2()
        self._block3 = self.block3()
        self._block4 = self.block4()
        if num_classes:
            self._logits = self.logits()
            self._softmax = self.softmax()

    def conv1(self):
        return Stride2BasicConvUnit(64, 7, name='conv1')

    def pool1(self):
        return tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='pool1')

    def block1(self):
        pass

    def block2(self):
        pass

    def block3(self):
        pass

    def block4(self):
        pass

    def logits(self):
        return tf.keras.layers.Conv2D(self._num_classes, 1, name='logits')

    def softmax(self):
        return tf.keras.layers.Softmax(name='softmax')

    def stack_blocks(self):
        blocks = [self._block1,
                  self._block2,
                  self._block3,
                  self._block4]
        return blocks

    def call(self, inputs, training=False):
        endpoints = OrderedDict()
        x = inputs
        x = self._conv1(x, training=training)
        endpoints[x.name[:-2]] = x
        x = self._pool1(x)
        endpoints[x.name[:-2]] = x
        for block in self.stack_blocks():
            x = block(x, training=training)
            endpoints[x.name[:-2]] = x
        if self._global_pool:
            x = tf.reduce_mean(x, axis=[1, 2], keepdims=True, name='pool5')
            endpoints[x.name[:-2]] = x
        if self._num_classes:
            x = self._logits(x)
            endpoints[x.name[:-2]] = x
            if self._spatial_squeeze:
                x = tf.squeeze(x, axis=[1, 2], name='SpatialSqueeze')
                endpoints[x.name[:-2]] = x
            endpoints['predictions'] = self._softmax(x)
        return x, endpoints

    def restore_from_checkpoint(self, checkpoint_dir):
        if not self.built:
            raise Exception('Build model first and restore variables.')
        checkpoint_variable_names, checkpoint_variable_shapes = zip(*tf.train.list_variables(checkpoint_dir))
        checkpoint_name_scope = _infer_name_scope(checkpoint_variable_names)
        model_name_scope = _infer_name_scope([variable.name for variable in self.variables])

        assignment_map = {}
        for variable in self.variables:
            converted_variable_name = _convert_name_to_slim_style(variable.name, model_name_scope, checkpoint_name_scope)
            if converted_variable_name in checkpoint_variable_names:
                assignment_map[converted_variable_name] = variable
        return tf.train.init_from_checkpoint(checkpoint_dir, assignment_map)


def _infer_name_scope(variable_names):
    weights_names = []
    for name in variable_names:
        if 'kernel' in name or 'weights' in name:
            weights_names.append(name)
    return os.path.commonpath(weights_names)


def _convert_name_to_slim_style(name, model_name_scope, checkpoint_name_scope):
    name = name.replace(model_name_scope, checkpoint_name_scope)
    if name.endswith(':0'):
        name = name[:-2]
    if name.endswith('conv2d/kernel'):
        name = name.replace('conv2d/kernel', 'weights')
    if name.endswith('logits/kernel'):
        name = name.replace('logits/kernel', 'logits/weights')
    if name.endswith('logits/bias'):
        name = name.replace('logits/bias', 'logits/biases')
    return name


class ResnetBlock(tf.keras.Model):
    def __init__(self, depth, num_units, strides, dtype=tf.float32, *args, **kwargs):
        super(ResnetBlock, self).__init__(*args, **kwargs)
        self._depth = depth
        self._num_units = num_units
        self._strides = strides
        self._dtype = dtype
        self._units = self.stack_units()

    def stack_units(self):
        units = []
        for i in range(self._num_units):
            bottleneck_class = BottleneckV1
            if i == self._num_units - 1 and self._strides == 2:
                bottleneck_class = Stride2BottleneckV1
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
        self._act = tf.keras.layers.Activation(activation, name='activation')

    def conv1(self):
        return BasicConvUnit(filters=self._depth, kernel_size=1, dtype=self._dtype, name='conv1')

    def conv2(self):
        return BasicConvUnit(filters=self._depth, kernel_size=3, dtype=self._dtype, name='conv2')

    def conv3(self):
        return BasicConvUnit(filters=4 * self._depth, kernel_size=1, activation=None, dtype=self._dtype, name='conv3')

    def shortcut(self):
        return lambda x: x

    def projection_shortcut(self):
        return BasicConvUnit(filters=4 * self._depth, kernel_size=1, activation=None, dtype=self._dtype,
                             name='shortcut')

    def build(self, input_shape):
        if input_shape[-1] == 4 * self._depth:
            self._shortcut = self.shortcut()
            self._projection_shortcut = False
        else:
            self._shortcut = self.projection_shortcut()
            self._projection_shortcut = True
        self.built = True

    def call(self, inputs, training=False):
        x = inputs
        x = self._conv1(x, training=training)
        x = self._conv2(x, training=training)
        x = self._conv3(x, training=training)
        if self._projection_shortcut:
            shortcut = self._shortcut(inputs, training=training)
        else:
            shortcut = self._shortcut(inputs)
        x = self._act(x + shortcut)
        return x


class Stride2BottleneckV1(BottleneckV1):
    def conv2(self):
        return Stride2BasicConvUnit(filters=self._depth, kernel_size=3, dtype=self._dtype, name='conv2')

    def projection_shortcut(self):
        return Stride2BasicConvUnit(filters=4 * self._depth, kernel_size=1, activation=None, dtype=self._dtype,
                                    name='shortcut')

    def shortcut(self):
        return tf.keras.layers.MaxPool2D(pool_size=1, strides=2, dtype=self._dtype, name='shortcut')


# Basic convolutional block. Conv2D + BatchNormalization + (ReLU)
class BasicConvUnit(tf.keras.Model):
    def __init__(self, filters, kernel_size, strides=1, padding='same', activation='relu', dtype=tf.float32,
                 *args, **kwargs):
        super(BasicConvUnit, self).__init__(*args, **kwargs)
        self._filters = filters
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._activation = activation
        self._dtype = dtype

        self.conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding, use_bias=False, dtype=dtype,
                                           name='conv2d')
        self.bn = tf.keras.layers.BatchNormalization(dtype=dtype, name='BatchNorm')  # TODO: freeze batch norm params
        if activation:
            self.act = tf.keras.layers.Activation(activation, dtype=dtype, name='activation')

    def call(self, inputs, training=False):
        x = inputs
        x = self.conv(x)
        x = self.bn(x, training=training)
        if self._activation:
            x = self.act(x)
        return x


class Stride2BasicConvUnit(BasicConvUnit):
    def __init__(self, filters, kernel_size, activation='relu', dtype=tf.float32, *args, **kwargs):
        strides = 2
        padding = 'valid'
        super(Stride2BasicConvUnit, self).__init__(filters, kernel_size, strides, padding, activation, dtype,
                                                   *args, **kwargs)

    def pad_inputs_for_valid_conv(self, inputs):
        pad_total = self._kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        # This means that pad inputs [[batch_beg, batch_end], [height_beg, ...], ..., [..., depth_end]] times
        # In case height or width is even, default Conv2D with strides=2 and padding='same' pads the inputs by
        # tf.pad(inputs, [[0, 0], [0, 1], [0, 1], [0, 0]) resulting into different outputs.
        return inputs

    def call(self, inputs, training=False):
        x = self.pad_inputs_for_valid_conv(inputs)
        x = super(Stride2BasicConvUnit, self).call(x, training=training)
        return x


class ResnetV1_50(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, 2, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 4, 2, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 6, 2, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, 1, dtype=self._dtype, name='block4')


class ResnetV1_101(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, 2, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 4, 2, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 23, 2, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, 1, dtype=self._dtype, name='block4')


class ResnetV1_152(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, 2, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 8, 2, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 36, 2, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, 1, dtype=self._dtype, name='block4')


class ResnetV1_200(ResnetV1):
    def block1(self):
        return ResnetBlock(64, 3, 2, dtype=self._dtype, name='block1')

    def block2(self):
        return ResnetBlock(128, 24, 2, dtype=self._dtype, name='block2')

    def block3(self):
        return ResnetBlock(256, 36, 2, dtype=self._dtype, name='block3')

    def block4(self):
        return ResnetBlock(512, 3, 1, dtype=self._dtype, name='block4')
