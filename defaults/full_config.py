from utils import Config

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

config = Config(
    encoding=encoding,
    central=central,
    decoding=decoding
)
