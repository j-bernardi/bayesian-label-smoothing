default_encoder = {
    "layer_1_2": {
        "num_filters_out": 16,
        "num_convs": 1,
    },
    "layer_2_3": {
        "num_filters_out": 32,
        "num_convs": 1,
    },
}

default_decoder = {
    "layer_3_2": {
        "num_filters_out": 32,
        "num_convs": 1,
    },
    "layer_2_1": {
        "num_filters_out": 16,
        "num_convs": 1,
    },
}

default_central = {
    "num_filters_out": 64,
        "num_convs": 1,
}