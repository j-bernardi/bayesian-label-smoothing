import tensorflow as tf

unet = True
mobile_net = False
encoding = {
    "layer_1_2": {
        "num_filters_out": 32,
        "kernel_size": 5,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },
    "layer_2_3": {
        "num_filters_out": 64,
        "kernel_size": 5,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },


    "layer_3_4": {
        "num_filters_out": 64,  # was 128
        "kernel_size": 5,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },


    "layer_4_5": {
        "num_filters_out": 128,
        "kernel_size": 3,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },
    "layer_5_6": {
        "num_filters_out": 128,
        "kernel_size": 3,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },
}


central = {
    "num_convs": 2,
    "num_filters_out": 128,
    "kernel_size": 3,
    "strides": 1
}


decoding = {
    "layer_6_5": {
        "num_filters_out": 128,
        "transpose_kernel_size": 5,
        "transpose_strides": 2,
        "num_convs": 0,
        # would have sep ks and stride ^
    },
    "layer_5_4": {
        "num_filters_out": 128,
        "transpose_kernel_size": 5,
        "transpose_strides": 2,
        "num_convs": 0,
    },
    "layer_4_3": {
        "num_filters_out": 64,
        "transpose_kernel_size": 5,
        "transpose_strides": 2,
        "num_convs": 0,
    },
    "layer_3_2": {
        "num_filters_out": 64,
        "transpose_kernel_size": 5,
        "transpose_strides": 2,
        "num_convs": 0,
    },

    "layer_2_1": {
        "num_filters_out": 32,
        "transpose_kernel_size": 5,
        "transpose_strides": 2,
        "num_convs": 0,
    },
}


max_epochs = 500
train_batch_size = 64  # they use 50
val_batch_size = 1000


# Overall % of data for training and valid
trn_split = 0.9
# Overalll % of data for valid, taken from training split
val_split = 0.15  # potentially 0.1
# Rest is test


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
