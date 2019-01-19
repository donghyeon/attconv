import tensorflow as tf
from official.transformer.model.attention_layer import SelfAttention
from research.object_detection.utils.shape_utils import combined_static_and_dynamic_shape


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
