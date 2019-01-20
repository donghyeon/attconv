import tensorflow as tf

from object_detection.models.ssd_resnet_v1_fpn_feature_extractor import _SSDResnetV1FpnFeatureExtractor
from object_detection.models import att_resnet_v1

slim = tf.contrib.slim


class _SSDAttResnetV1FpnFeatureExtractor(_SSDResnetV1FpnFeatureExtractor):
    def __init__(self, is_training, *args,
                 num_layers=1, num_heads=2, attention_dropout=0.1, **kwargs):
        super(_SSDAttResnetV1FpnFeatureExtractor, self).__init__(
            is_training, *args, **kwargs)
        self._is_training = is_training
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout

    def extract_features(self, preprocessed_inputs):
        train = self._is_training  # This will enable dropout and LayerNormalization on attention layers only.
        with slim.arg_scope([att_resnet_v1.self_attention], train=train):
            return super(_SSDAttResnetV1FpnFeatureExtractor, self).extract_features(preprocessed_inputs)

    def _filter_features(self, image_features):
        filtered_image_features = super(_SSDAttResnetV1FpnFeatureExtractor, self)._filter_features(image_features)
        for key, feature in image_features.items():
            if key.endswidth('attention'):
                block = key.split('/')[-2]
                filtered_image_features[block] = feature
        return filtered_image_features

    def restore_from_classification_checkpoint_fn(self, feature_extractor_scope):
        variables_to_restore = {}
        for variable in tf.global_variables():
            var_name = variable.op.name
            if var_name.startswith(feature_extractor_scope + '/' + 'att_'):
                var_name = var_name.replace(feature_extractor_scope + '/' + 'att_', '')
                variables_to_restore[var_name] = variable
        return variables_to_restore


class SSDAttResnet50V1FpnFeatureExtractor(_SSDAttResnetV1FpnFeatureExtractor):
    """SSD AttResnet50 V1 FPN feature extractor."""
    def __init__(self, *args, **kwargs):
        super(SSDAttResnet50V1FpnFeatureExtractor, self).__init__(
            *args, resnet_base_fn=att_resnet_v1.att_resnet_v1_50,
            resnet_scope_name='att_resnet_v1_50',
            fpn_scope_name='fpn', **kwargs)


class SSDAttResnet101V1FpnFeatureExtractor(_SSDAttResnetV1FpnFeatureExtractor):
    """SSD AttResnet101 V1 FPN feature extractor."""
    def __init__(self, *args, **kwargs):
        super(SSDAttResnet101V1FpnFeatureExtractor, self).__init__(
            *args, resnet_base_fn=att_resnet_v1.att_resnet_v1_101,
            resnet_scope_name='att_resnet_v1_101',
            fpn_scope_name='fpn', **kwargs)


class SSDAttResnet152V1FpnFeatureExtractor(_SSDAttResnetV1FpnFeatureExtractor):
    """SSD AttResnet152 V1 FPN feature extractor."""
    def __init__(self, *args, **kwargs):
        super(SSDAttResnet152V1FpnFeatureExtractor, self).__init__(
            *args, resnet_base_fn=att_resnet_v1.att_resnet_v1_152,
            resnet_scope_name='att_resnet_v1_152',
            fpn_scope_name='fpn', **kwargs)
