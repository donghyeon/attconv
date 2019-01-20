import tensorflow as tf

from object_detection.models.faster_rcnn_resnet_v1_feature_extractor import FasterRCNNResnetV1FeatureExtractor
from object_detection.models import att_resnet_v1

slim = tf.contrib.slim


class FasterRCNNAttResnetV1FeatureExtractor(FasterRCNNResnetV1FeatureExtractor):
    def __init__(self, architecture, resnet_model, is_training, *args,
                 num_layers=1, num_heads=2, attention_dropout=0.1, **kwargs):
        super(FasterRCNNAttResnetV1FeatureExtractor, self).__init__(
            architecture, resnet_model, is_training, *args, **kwargs)
        self._is_training = is_training
        self._num_layers = num_layers
        self._num_heads = num_heads
        self._attention_dropout = attention_dropout

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        train = self._is_training  # This will enable dropout and LayerNormalization on attention layers only.
        with slim.arg_scope([att_resnet_v1.self_attention], train=train):
            _, activations = super(FasterRCNNAttResnetV1FeatureExtractor, self)._extract_proposal_features(
                preprocessed_inputs, scope)
        handle = scope + '/%s/block3/attention' % self._architecture
        return activations[handle], activations

    def restore_from_classification_checkpoint_fn(
            self, first_stage_feature_extractor_scope, second_stage_feature_extractor_scope):
        variables_to_restore = {}
        for variable in tf.global_variables():
            for scope_name in [first_stage_feature_extractor_scope,
                               second_stage_feature_extractor_scope]:
                if variable.op.name.startswith(scope_name + '/' + 'att_'):
                    var_name = variable.op.name.replace(scope_name + '/' + 'att_', '')
                    variables_to_restore[var_name] = variable
        return variables_to_restore


class FasterRCNNAttResnet50FeatureExtractor(FasterRCNNAttResnetV1FeatureExtractor):
    def __init__(self, *args, **kwargs):
        super(FasterRCNNAttResnet50FeatureExtractor, self).__init__(
            'att_resnet_v1_50', att_resnet_v1.att_resnet_v1_50, *args, **kwargs)


class FasterRCNNAttResnet101FeatureExtractor(FasterRCNNAttResnetV1FeatureExtractor):
    def __init__(self, *args, **kwargs):
        super(FasterRCNNAttResnet101FeatureExtractor, self).__init__(
            'att_resnet_v1_101', att_resnet_v1.att_resnet_v1_101, *args, **kwargs)


class FasterRCNNAttResnet152FeatureExtractor(FasterRCNNAttResnetV1FeatureExtractor):
    def __init__(self, *args, **kwargs):
        super(FasterRCNNAttResnet152FeatureExtractor, self).__init__(
            'att_resnet_v1_152', att_resnet_v1.att_resnet_v1_152, *args, **kwargs)
