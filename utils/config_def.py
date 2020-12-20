import tensorflow as tf

from defaults.sample_model_layers import (
    default_encoder, default_central, default_decoder
)


class Config(object):

    def __init__(self, **kwargs):

        self.unet = kwargs.pop("unet_flag", True)  # only option

        self.encoding = kwargs.pop("encoding", default_encoder)
        self.central = kwargs.pop("central", default_central)
        self.decoding = kwargs.pop("decoding", default_decoder)

        self.max_epochs = kwargs.pop("max_epochs", 500)

        self.train_batch_size = kwargs.pop("train_batch_size", 64)
        self.val_batch_size = kwargs.pop("val_batch_size", 1000)

        # Overall % of data for training AND valid (rest test)
        self.train_split = kwargs.pop("train_split", 0.9)
        # Overalll % of data for valid, taken from training split
        self.val_split = kwargs.pop("val_split", 0.15)

        self.optim = kwargs.pop("optim",
            tf.keras.optimizers.Adam(learning_rate=0.001)
        )
        self.loss_args = kwargs.pop("loss_args", {})

        self.callback_args = kwargs.pop("callback_args",
            {"es_delta": 0.0001, "es_patience": 8, "rlr_factor": 0.33,
            "rlr_patience": 4, "rlr_delta": 0.001, "rlr_min": 0.00001,
            }
        )

        self.class_weight_mode = kwargs.pop("class_weight_mode", "uniform")
        self.drop_background = kwargs.pop("drop_background", None)

        self.smoothing_function = kwargs.pop("smoothing_function",
            self.default_smoothing_function
        )

        if kwargs:
            raise ValueError(f"{kwargs.keys()} weren't read")

    @staticmethod
    def default_smoothing_function(  # or None
        n_classes, training_generator, num_training_batches
    ):
            return 0  # NO SMOOTHING

            # Else adapt function and args as desired
            return fixed_adjacent_smoothing(
                n_classes=n_classes,
                training_generator=training_generator,
                fixed_smoothing_magnitude=0.9,
                generator_length=num_training_batches
            )
