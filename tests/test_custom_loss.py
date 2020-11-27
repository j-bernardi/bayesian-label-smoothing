import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy

from utils import GLOBAL_TYPE
from losses import CustomSmoothedWeightedCCE

tf.keras.backend.set_floatx(GLOBAL_TYPE)


# TODO do tests for smoothing as well - uniform should be easy
# and both together maybe

def test_weights_end_to_end():

    y_true = np.zeros((1, 2, 1, 11))
    y_true[0, 0, 0, 0] = 1
    y_true[0, 1, 0, 10] = 1
    
    y_pred = np.random.rand(1, 2, 1, 11)
    
    weights = [1. for _ in range(11)]
    weights[0] = 0.01
    weights[10] = 100.
    
    ls = CustomSmoothedWeightedCCE(
        class_weights=weights, from_logits=True
    )
    loss = ls(y_true, y_pred)


def test_weights_zeros():

    y_true = np.zeros((1, 2, 1, 11))
    y_true[0, 0, 0, 0] = 1
    y_true[0, 1, 0, 10] = 1
    
    y_pred = np.random.rand(1, 2, 1, 11)
    
    weights = [0. for _ in range(11)]
    
    ls = CustomSmoothedWeightedCCE(
        class_weights=weights, from_logits=True
    )
    loss = ls(y_true, y_pred)

    assert loss == 0.


def test_weights_reproduce():

    y_true = np.zeros((1, 2, 1, 11))
    y_true[0, 0, 0, 0] = 1
    y_true[0, 1, 0, 10] = 1
    
    y_pred = np.random.rand(1, 2, 1, 11)
    
    weights = [1. for _ in range(11)]
    weights_2 = [2. for _ in range(11)]

    ls = CustomSmoothedWeightedCCE(
        class_weights=weights, from_logits=True
    )
    ls_2 = CustomSmoothedWeightedCCE(
        class_weights=weights_2, from_logits=True
    )
    original_ls = CategoricalCrossentropy(from_logits=True)
    
    loss = tf.cast(ls(y_true, y_pred), GLOBAL_TYPE)
    loss_2 = tf.cast(ls_2(y_true, y_pred), GLOBAL_TYPE)
    original_loss = tf.cast(original_ls(y_true, y_pred), GLOBAL_TYPE)

    assert original_loss == loss
    assert 2. * original_loss == loss_2


def test_smoothing_reproduce():
    n = 10
    smooth_mag = 0.9
    # NOTE: there is a bug in TF2.1 (fixed in 2.3) where
    # num_classes is read from the [1] dimension, typo
    # for [-1]. Hacking here for test pass
    y_true = np.zeros((1, n, 2, n), dtype=GLOBAL_TYPE)
    y_true[0, :, 0, 0] = 1
    y_true[0, :, 1:, n-1] = 1

    # This is how tf implements smoothing
    y_smoothed = np.zeros((1, n, 2, n), dtype=GLOBAL_TYPE) + (1.-smooth_mag) / n
    y_smoothed[0, :, 0, 0] = smooth_mag + (1.-smooth_mag) / n
    y_smoothed[0, :, 1:, n-1] = smooth_mag + (1.-smooth_mag) / n
    
    y_pred = tf.convert_to_tensor(
        np.random.rand(1, n, 2, n),
        dtype=GLOBAL_TYPE
    )
    # Should be same if implemented as matrix
    smoothing = np.zeros((n, n)) + (1.-smooth_mag) / n
    for c in range(n):
        # This(+=) iscurrently how TF operates.
        smoothing[c, c] += smooth_mag
    ls = CustomSmoothedWeightedCCE(
        from_logits=True, label_smoothing=smoothing,
    )
    original_ls = CategoricalCrossentropy(
        from_logits=True, label_smoothing=(1.-smooth_mag)
    )
    original_ls_clean = CategoricalCrossentropy(from_logits=True, label_smoothing=0)

    loss = tf.cast(ls(y_true, y_pred), GLOBAL_TYPE)
    original_loss = tf.cast(original_ls(y_true, y_pred), GLOBAL_TYPE)
    original_clean = tf.cast(original_ls_clean(y_smoothed, y_pred), GLOBAL_TYPE)

    # a tf test but good for sanity on the following
    assert np.all(np.isclose(original_loss, original_clean))
    assert np.all(np.isclose(original_loss, loss))


if __name__ == "__main__":
    # Lazy testing - TODO pytest
    test_weights_end_to_end()
    test_weights_zeros()
    test_weights_reproduce()
    test_smoothing_reproduce()
    print("Pass")
