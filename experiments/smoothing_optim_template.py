from utils import Config
from losses import (
    fixed_uniform_smoothing,
    fixed_adjacent_smoothing,
    weighted_uniform_smoothing,
    weighted_adjacent_smoothing,
)


def smoothing_function(  # or smoothing_function = None
    n_classes, training_generator, num_training_batches
):
    if "FUNCTION_PLACEHOLDER" == "fixed_uniform_smoothing":
        return fixed_uniform_smoothing(
            n_classes,
            PARAMETER_PLACEHOLDER,
        )
    else:
        return FUNCTION_PLACEHOLDER(
            n_classes,
            training_generator,
            PARAMETER_PLACEHOLDER,
            generator_length=num_training_batches
        )


encoding = {
    "layer_1_2": {
        "num_filters_out": 32,
        "num_convs": 2,
        "kernel_size": 5,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },

    "layer_2_3": {
        "num_filters_out": 64,
        "num_convs": 2,
        "kernel_size": 5,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },

    "layer_3_4": {
        "num_filters_out": 64,  # was 128
        "num_convs": 2,
        "kernel_size": 5,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },

    "layer_4_5": {
        "num_filters_out": 128,
        "num_convs": 2,
        "kernel_size": 3,
        "strides": 1,
        "pool_size": 3,
        "pool_strides": 2,
    },

    "layer_5_6": {
        "num_filters_out": 128,
        "num_convs": 2,
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
        "num_filters_out": 32,  # was n class
        "transpose_kernel_size": 5,
        "transpose_strides": 2,
        "num_convs": 0,
    },
}


config = Config(
    encoding=encoding,
    central=central,
    decoding=decoding,
    smoothing_function=smoothing_function,
)
