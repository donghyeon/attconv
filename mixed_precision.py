import tensorflow as tf


class MixedPrecisionLossScaleOptimizer(tf.contrib.mixed_precision.LossScaleOptimizer):
    def __init__(self, *args, **kwargs):
        self.built = False
        super(MixedPrecisionLossScaleOptimizer, self).__init__(*args, **kwargs)

    def build_fp32_variables(self, variables):
        vars_fp16_to_fp32 = {}
        vars_fp32_to_fp16 = {}
        for var in variables:
            if var.dtype == tf.float16:
                name = var.name.split(':')[0] + '_fp32'
                var_fp32 = tf.Variable(
                    initial_value=tf.cast(var.initialized_value(), dtype=tf.float32),
                    name=name,
                    expected_shape=var.shape,
                    dtype=tf.float32,
                    trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 "FP32_MASTER_COPIES"])
                vars_fp16_to_fp32[var] = var_fp32
                vars_fp32_to_fp16[var_fp32] = var
        self.vars_fp16_to_fp32 = vars_fp16_to_fp32
        self.vars_fp32_to_fp16 = vars_fp32_to_fp16
        self.built = True

    def compute_gradients(self, loss, *args, **kwargs):
        loss_scale = self._loss_scale_manager.get_loss_scale()
        if tf.executing_eagerly():
            def scaled_loss():
                loss_val = loss()
                return loss_val * tf.cast(loss_scale, loss_val.dtype.base_dtype)
        else:
            if callable(loss):
                loss_val = loss()
            else:
                loss_val = loss
            scaled_loss = loss_val * tf.cast(loss_scale, loss_val.dtype.base_dtype)
        grads_and_vars = self._opt.compute_gradients(scaled_loss, *args, **kwargs)
        if not self.built:
            gradients, variables = zip(*grads_and_vars)
            self.build_fp32_variables(variables)
        grads_and_vars_fp32 = self._cast_fp32_and_down_scale(grads_and_vars, loss_scale)
        return grads_and_vars_fp32

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        update_op = super(MixedPrecisionLossScaleOptimizer, self).apply_gradients(grads_and_vars, *args, **kwargs)
        assign_fp32_to_fp16_ops = []
        with tf.control_dependencies([update_op]):
            for grad, var in grads_and_vars:
                if var in self.vars_fp32_to_fp16:
                    var_fp16 = self.vars_fp32_to_fp16[var]
                    assign_op = tf.assign(var_fp16, tf.saturate_cast(var, tf.float16))
                    assign_fp32_to_fp16_ops.append(assign_op)
        if assign_fp32_to_fp16_ops:
            return tf.group(assign_fp32_to_fp16_ops)
        return update_op

    def _cast_fp32_and_down_scale(self, grads_and_vars, loss_scale):
        # Down scale grads by the loss_scale.
        grads_and_vars_fp32 = []
        inv_loss_scale = tf.cast(tf.math.reciprocal(loss_scale), tf.float32)
        for grad, var in grads_and_vars:
            if var.dtype == tf.float16:
                if grad is not None:
                    grad = tf.cast(grad, tf.float32) * inv_loss_scale
                    var = self.vars_fp16_to_fp32[var]
            grads_and_vars_fp32.append((grad, var))
        return grads_and_vars_fp32
