import tensorflow as tf
from losses import (
    fixed_uniform_smoothing,
    fixed_adjacent_smoothing,
    weighted_uniform_smoothing,
    weighted_adjacent_smoothing,
)

encoding = {
    "layer_1_2": {
        "num_filters_out": 16,
         "num_convs": 1,
    },
    "layer_2_3": {
        "num_filters_out": 32,
        "num_convs": 1,
    },
}

decoding = {
    "layer_3_2": {
        "num_filters_out": 32,
        "num_convs": 1,
    },
    "layer_2_1": {
        "num_filters_out": 16,
        "num_convs": 1,
    },
}

max_epochs = 500
train_batch_size = 64
val_batch_size = 1000

# Overall % of data for training and valid
trn_split = 0.8
# Overalll % of data for valid, taken from training split
val_split = 0.2

class_weight_mode = "uniform"
drop_background = None
def smoothing_function(  # or smoothing_function = None
    n_classes, training_generator, num_training_batches
):
    # Adapt function and args as desired
    return fixed_adjacent_smoothing(
        n_classes=n_classes,
        training_generator=training_generator,
        fixed_smoothing_magnitude=0.9,
        generator_length=num_training_batches
    )

optim = tf.keras.optimizers.Adam(
    learning_rate=0.001,
)

loss_args = {
    # "label_smoothing": 0.1,
}

callback_args = {
    "es_delta": 0.0001,
    "es_patience": 8,
    "rlr_factor": 0.33,
    "rlr_patience": 4,
    "rlr_delta": 0.001,
    "rlr_min": 0.00001,
}
