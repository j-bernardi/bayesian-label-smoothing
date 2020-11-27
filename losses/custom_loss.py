import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils

from utils import GLOBAL_TYPE

tf.keras.backend.set_floatx(GLOBAL_TYPE)


class CustomSmoothedWeightedCCE(object):
    """Custom CCE implementing label smoothing, class weighting

    Categorical Crossentropy loss function with custom
    implementation for label smoothing and 
    """

    def __init__(
        self,
        class_weights=None,
        from_logits=False,
        label_smoothing=0,
        name="custom_weighted_cce"
    ):
        """
        Args:
            class_weights:

            label_smoothing (int|np.ndarray): int for uniform
                label smoothing (as defined in keras CCE), or
                a numpy array
        """
        if class_weights is None or all(v == 1. for v in class_weights):
            self.class_weights = None
        else:
            self.class_weights = tf.convert_to_tensor(
                class_weights, dtype=GLOBAL_TYPE
            )

        if (
            isinstance(label_smoothing, float) 
            or (isinstance(label_smoothing, int)
                and label_smoothing in (0, 1))
        ):
            if not label_smoothing == 0:
                raise NotImplementedError(
                    "There is a bug in TF2.1 (fixed in 2.3) where"
                    "num_classes is read from the [1] dimension, typo"
                    "for [-1]. Keeping unimplemented here."
                )
            smoothing_arg = label_smoothing
            self.label_smoothing = None
        elif label_smoothing.ndim == 2:
            smoothing_arg = 0
            self.label_smoothing = tf.convert_to_tensor(
                label_smoothing, dtype=GLOBAL_TYPE
            )
        else:
            raise ValueError(
                f"Smoothing must be integer (uniform) or (n_class, n_class)"
                f"array. Got {label_smoothing}"
            )

        self.unreduced_cce_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=from_logits,
            label_smoothing=smoothing_arg,
            reduction=losses_utils.ReductionV2.NONE,
            name=name,
        )


    def __call__(self, y_true, y_pred, sample_weight=None):

        if sample_weight is not None:
            raise NotImplementedError(
                f"This loss function is only for implementing class weights."
                f" sample_weight not None ({sample_weight}) not valid."
            )

        true_classes = None
        # Replace one-hot with smoothed labels
        if self.label_smoothing is not None:
            true_classes = tf.argmax(y_true, axis=-1)
            y_true = tf.gather(self.label_smoothing, true_classes)

        unreduced_loss = tf.dtypes.cast(
            self.unreduced_cce_fn(y_true, y_pred),
            GLOBAL_TYPE
        )

        # Weight the losses
        if self.class_weights is not None:
            true_classes = tf.argmax(y_true, axis=-1) if true_classes is None else true_classes
            weight_mask = tf.gather(self.class_weights, true_classes)
            loss = tf.math.multiply(unreduced_loss, weight_mask)
        else:
            loss = unreduced_loss

        return losses_utils.reduce_weighted_loss(
            loss, reduction=losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
        )
