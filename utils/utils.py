import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

from .globals import GLOBAL_TYPE

tf.keras.backend.set_floatx(GLOBAL_TYPE)


class CustomWeightedCCE(object):

    def __init__(self, class_weights, from_logits=False, label_smoothing=0, name="custom_weighted_cce"):
        self.unreduced_cce_fn = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing,
            from_logits=from_logits,
            reduction=losses_utils.ReductionV2.NONE,  # key
            name=name,
        )

        self.class_weights = tf.convert_to_tensor(class_weights, dtype=GLOBAL_TYPE)

    def __call__(self, y_true, y_pred, sample_weight=None):

        if sample_weight is not None:
            raise NotImplementedError(
                f"This loss function is only for implementing class weights."
                f" sample_weight not None ({sample_weight}) not valid."
            )
        unreduced_loss = tf.dtypes.cast(
            self.unreduced_cce_fn(y_true, y_pred),
            GLOBAL_TYPE
        )

        true_classes = tf.argmax(y_true, axis=-1)
        weight_mask = tf.gather(self.class_weights, true_classes)

        return losses_utils.reduce_weighted_loss(
            tf.math.multiply(unreduced_loss, weight_mask),
            reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
        )
