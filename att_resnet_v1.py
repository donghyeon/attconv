import tensorflow as tf
from research.slim.nets import resnet_v1
from research.slim.nets import resnet_utils
from official.transformer.model.attention_layer import SelfAttention
from research.object_detection.utils.shape_utils import combined_static_and_dynamic_shape

slim = tf.contrib.slim
bottleneck = resnet_v1.bottleneck
NoOpScope = resnet_v1.NoOpScope
subsample = resnet_utils.subsample
resnet_v1_block = resnet_v1.resnet_v1_block


class SelfAttention2D(SelfAttention):
    def __init__(self, hidden_size, num_heads, attention_dropout, train):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train
        super(SelfAttention2D, self).__init__(hidden_size, num_heads, attention_dropout, train)

    def call(self, x):
        batch_size, width, height, num_channels = combined_static_and_dynamic_shape(x)
        num_pixels = width * height
        reshape_to_1d_layer = tf.keras.layers.Reshape([num_pixels, num_channels])
        reshape_to_2d_layer = tf.keras.layers.Reshape([width, height, self.hidden_size])

        x = reshape_to_1d_layer(x)
        x = super(SelfAttention2D, self).call(x, bias=(
            tf.zeros([batch_size, 1, 1, num_pixels])))  # No bias needed for attention on conv feature maps
        x = reshape_to_2d_layer(x)
        return x


@slim.add_arg_scope
def self_attention(inputs, hidden_size, num_heads, attention_dropout, is_training,
                   outputs_collections=None, scope=None, **kwargs):
    with tf.variable_scope(scope, 'self_attention', [inputs]) as sc:
        outputs = SelfAttention2D(hidden_size, num_heads, attention_dropout, is_training)(inputs)
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, outputs)


def att_resnet_v1(inputs,
                  blocks,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  include_root_block=True,
                  spatial_squeeze=True,
                  store_non_strided_activations=False,
                  reuse=None,
                  scope=None):
    with tf.variable_scope(scope, 'att_resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             stack_attention_blocks_dense],
                            outputs_collections=end_points_collection):
            with (slim.arg_scope([slim.batch_norm, self_attention], is_training=is_training)
                    if is_training is not None else NoOpScope()):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = stack_attention_blocks_dense(net, blocks, output_stride,
                                                   store_non_strided_activations)
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return net, end_points
att_resnet_v1.default_image_size = 224


def attention_block(scope, num_layers, hidden_size, num_heads, attention_dropout, stride):
    return resnet_utils.Block(scope, self_attention, [{
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'attention_dropout': attention_dropout,
        'stride': 1
    }] * (num_layers - 1) + [{
        'hidden_size': hidden_size,
        'num_heads': num_heads,
        'attention_dropout': attention_dropout,
        'stride': stride
    }])


@slim.add_arg_scope
def stack_attention_blocks_dense(net, blocks, output_stride=None,
                                 store_non_strided_activations=False,
                                 outputs_collections=None):
    current_stride = 1
    rate = 1

    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for i, unit in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    # Move stride from the block's last unit to the end of the block.
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)

                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)

                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')

            # Collect activations at the block's end before performing subsampling.
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

            # Subsampling of the block's output activations.
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')

    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')

    return net


def att_resnet_v1_50(inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     output_stride=None,
                     spatial_squeeze=True,
                     store_non_strided_activations=False,
                     reuse=None,
                     scope='att_resnet_v1_50'):
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_block('block3_pre_att', base_depth=256, num_units=6, stride=1),
        attention_block('block3', num_layers=1, hidden_size=1024, num_heads=2, attention_dropout=0.1, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return att_resnet_v1(inputs, blocks, num_classes, is_training,
                         global_pool=global_pool, output_stride=output_stride,
                         include_root_block=True, spatial_squeeze=spatial_squeeze,
                         store_non_strided_activations=store_non_strided_activations,
                         reuse=reuse, scope=scope)
att_resnet_v1_50.default_image_size = att_resnet_v1.default_image_size


def att_resnet_v1_101(inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      spatial_squeeze=True,
                      store_non_strided_activations=False,
                      reuse=None,
                      scope='att_resnet_v1_101'):
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_block('block3_pre_att', base_depth=256, num_units=23, stride=1),
        attention_block('block3', num_layers=1, hidden_size=1024, num_heads=2, attention_dropout=0.1, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return att_resnet_v1(inputs, blocks, num_classes, is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       store_non_strided_activations=store_non_strided_activations,
                       reuse=reuse, scope=scope)
att_resnet_v1_101.default_image_size = att_resnet_v1.default_image_size


def att_resnet_v1_152(inputs,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  store_non_strided_activations=False,
                  spatial_squeeze=True,
                  reuse=None,
                  scope='att_resnet_v1_152'):
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=8, stride=2),
        resnet_v1_block('block3_pre_att', base_depth=256, num_units=36, stride=1),
        attention_block('block3', num_layers=1, hidden_size=1024, num_heads=2, attention_dropout=0.1, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return att_resnet_v1(inputs, blocks, num_classes, is_training,
                         global_pool=global_pool, output_stride=output_stride,
                         include_root_block=True, spatial_squeeze=spatial_squeeze,
                         store_non_strided_activations=store_non_strided_activations,
                         reuse=reuse, scope=scope)
att_resnet_v1_152.default_image_size = att_resnet_v1.default_image_size


def att_resnet_v1_200(inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      store_non_strided_activations=False,
                      spatial_squeeze=True,
                      reuse=None,
                      scope='att_resnet_v1_200'):
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=24, stride=2),
        resnet_v1_block('block3_pre_att', base_depth=256, num_units=36, stride=1),
        attention_block('block3', num_layers=1, hidden_size=1024, num_heads=2, attention_dropout=0.1, stride=2),
        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
    ]
    return att_resnet_v1(inputs, blocks, num_classes, is_training,
                         global_pool=global_pool, output_stride=output_stride,
                         include_root_block=True, spatial_squeeze=spatial_squeeze,
                         store_non_strided_activations=store_non_strided_activations,
                         reuse=reuse, scope=scope)
att_resnet_v1_200.default_image_size = att_resnet_v1.default_image_size