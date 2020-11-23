import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from utils import CustomWeightedCCE,  GLOBAL_TYPE

tf.keras.backend.set_floatx(GLOBAL_TYPE)


def test_end_to_end():

    y_true = np.zeros((1, 2, 1, 11))
    y_true[0, 0, 0, 0] = 1
    y_true[0, 1, 0, 10] = 1
    
    y_pred = np.random.rand(1, 2, 1, 11)
    
    weights = [1. for _ in range(11)]
    weights[0] = 0.01
    weights[10] = 100.
    
    ls = CustomWeightedCCE(weights, from_logits=True)
    loss = ls(y_true, y_pred)

    print(loss)


def test_zeros():

    y_true = np.zeros((1, 2, 1, 11))
    y_true[0, 0, 0, 0] = 1
    y_true[0, 1, 0, 10] = 1
    
    y_pred = np.random.rand(1, 2, 1, 11)
    
    weights = [0. for _ in range(11)]
    
    ls = CustomWeightedCCE(weights, from_logits=True)
    loss = ls(y_true, y_pred)

    assert loss == 0.


def test_reproduce():

    y_true = np.zeros((1, 2, 1, 11))
    y_true[0, 0, 0, 0] = 1
    y_true[0, 1, 0, 10] = 1
    
    y_pred = np.random.rand(1, 2, 1, 11)
    
    weights = [1. for _ in range(11)]
    weights_2 = [2. for _ in range(11)]

    ls = CustomWeightedCCE(weights, from_logits=True)
    ls_2 = CustomWeightedCCE(weights_2, from_logits=True)
    original_ls = CategoricalCrossentropy(from_logits=True)
    
    loss = tf.cast(ls(y_true, y_pred), GLOBAL_TYPE)
    loss_2 = tf.cast(ls_2(y_true, y_pred), GLOBAL_TYPE)
    original_loss = tf.cast(original_ls(y_true, y_pred), GLOBAL_TYPE)

    assert original_loss == loss
    assert 2. * original_loss == loss_2


if __name__ == "__main__":
    # Lazy testing - TODO pytest
    test_end_to_end()
    test_zeros()
    test_reproduce()
