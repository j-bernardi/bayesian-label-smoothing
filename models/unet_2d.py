import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
from collections import deque

from models.blocks import EncodeBlock, DecodeBlock


class UNet2D(tf.keras.Model):
 
    def __init__(
        self, input_shape, n_classes,
        encoding=None, decoding=None,
        central=None,
    ):
        """
        Args:
            input_shape: channels_last, excluding batch
                dimension, e.g. height, width, channels
            n_classes: number of output classes required
            encoding: dict of name: argdicts for encoding
                blocks
            decoding: dict of name: argdicts for decoding
                blocks

        """

        super(UNet2D, self).__init__()

        ### TODO ###
        # Adding one FC at the bottom
        #   Experiment with 2x conv, 1x conv at each layer
        #   Adding another level of depth (and experiment 
        #   with 2x conv, 1x conv...)
        ############
        self.unbuilt = True

        # Defaults
        if not encoding:
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
        if not decoding:
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
        if not central:
            central = {
                "num_filters_out": 64,
                "num_convs": 1,
            }

        self.encoders = []
        self.decoders = []
        encoded_shapes = [  # Keep track for upsampling padding
            np.array(
                [input_shape[0], input_shape[1]],
                dtype=np.int32
            )
        ]

        for nm, argdict in encoding.items():
            self.encoders.append(
                EncodeBlock(**argdict, name=nm)
            )
            # TODO - generalise for whatever in argdict
            encoded_shapes.append(
                np.ceil(encoded_shapes[-1] / 2.).astype(np.int32)
            )
        print("Expected encoded shapes")
        for es in encoded_shapes:
            print(es)

        # BOTTOM LAYER, consider batchnorms..
        self.bottom_block = []
        for n in range(central["num_convs"]):
            self.bottom_block.append(
                kl.Conv2D(  # Encode at bottom layer
                    central["num_filters_out"], (2, 2),
                    strides=1, padding="same",
                    activation=None,
                    name=f"bottom_{n}",
                )
            )
            self.bottom_block.append(kl.LeakyReLU())
            
        encoded_shapes.pop(-1)  # Bottom shape not relevant
        # Consider flatten and fully connected here
        for nm, argdict in decoding.items():
            self.decoders.append(
                DecodeBlock(
                    **argdict, name=nm,
                    crop_to_shape=encoded_shapes.pop(-1)
                )
            )

        # OUT
        self.out_layer = kl.Conv2D(  # outputs
            n_classes, kernel_size=1, strides=1,
            padding="same", name="last",
        )

        # TODO consider fc layer out?

    def call(self, x):

        encodeds = deque(maxlen=len(self.encoders))
        for encoder in self.encoders:
            x, encoded = encoder(x)
            encodeds.append(encoded)

        if self.unbuilt:
            print("Bottom layer input", x.shape)

        for action in self.bottom_block:
            x = action(x)

        if self.unbuilt:
            print("Bottom layer output", x.shape)

        for decoder in self.decoders:
            x = decoder(x, encodeds.pop())
        assert not encodeds
        self.unbuilt = False

        x = self.out_layer(x)

        if self.unbuilt:
            print("Output", x.shape)        
        return x
