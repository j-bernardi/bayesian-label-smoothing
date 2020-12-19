from losses import (
    fixed_uniform_smoothing,
    fixed_adjacent_smoothing,
    weighted_uniform_smoothing,
    weighted_adjacent_smoothing,
)


class_weight_mode = "uniform"

drop_background = None

def smoothing_function(  # or smoothing_function = None
    n_classes, training_generator, num_training_batches
):
    # return 0

    # Adapt function and args as desired
    return fixed_adjacent_smoothing(
        n_classes=n_classes,
        training_generator=training_generator,
        fixed_smoothing_magnitude=0.9,
        generator_length=num_training_batches
    )


config = Config(
    class_weight_mode=class_weight_mode,
    drop_background=drop_background,
    smoothing_function=smoothing_function,
)
