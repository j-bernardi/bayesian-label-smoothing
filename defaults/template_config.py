import tensorflow as tf


unet = True
mobile_net = False
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

max_epochs = 1  # 500
train_batch_size = 64
val_batch_size = 1000

# Overall % of data for training and valid
trn_split = 0.8
# Overalll % of data for valid, taken from training split
val_split = 0.2

optim = tf.keras.optimizers.Adam(
    learning_rate=0.001,
)

loss = tf.keras.losses.CategoricalCrossentropy(
    # label_smoothing=0.1  # add custom here
    from_logits=True,
)

callback_args = {
    "es_delta": 0.0001,
    "es_patience": 8,
    "rlr_factor": 0.33,
    "rlr_patience": 4,
    "rlr_delta": 0.001,
    "rlr_min": 0.00001,
}
