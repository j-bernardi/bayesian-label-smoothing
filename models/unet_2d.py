import tensorflow as tf
import tensorflow.keras.layers as kl
from collections import deque


class DecodeBlock(tf.keras.layers.Layer):
    """Layer to decode a UNet

    """

    def __init__(
        self, num_filters_out, num_convs=1, name="decode",
        batchnorm_every=False, batchnorm_last=False,
    ):
        """
        Args:
            num_filters_out: number of filters in
                the layer we're decoding *to*
        """
        super(DecodeBlock, self).__init__(name=name)

        self.upsample = kl.Conv2DTranspose(
            num_filters_out, kernel_size=(2, 2), strides=2, 
            padding='same', activation="relu",
            data_format=None, kernel_initializer='he_normal',
        )
        self.concat = kl.Concatenate(axis=-1)

        self.convs = []
        for n in range(num_convs):
            self.convs.append(
                kl.Conv2D(
                    num_filters_out, kernel_size=(2, 2),
                    strides=1, padding="same",
                    activation="relu", kernel_initializer='he_normal',
                )
            )
            if batchnorm_every:
                self.convs.append(kl.BatchNormalization())
        if batchnorm_last and not batchnorm_every:
            self.convs.append(kl.BatchNormalization())

    def call(self, x, tensor_for_concat):

        x = self.upsample(x)
        x = self.concat((tensor_for_concat, x))
        for conv in self.convs:
            x = conv(x)
        return x


class EncodeBlock(tf.keras.layers.Layer):
    """Layer to encode a UNet

    Returns the output as well as the 
    skip-connection tensor
    """

    def __init__(
        self, num_filters_out, num_convs=1, name="encode",
        batchnorm_every=False, batchnorm_last=True,
    ):
        """
        Args:
            num_filters_out: number of filters in
                the layer we're decoding *to*
        """
        super(EncodeBlock, self).__init__(name=name)

        self.convs = []
        for n in range(num_convs):
            self.convs.append(
                kl.Conv2D(
                    filters=num_filters_out, 
                    kernel_size=(2, 2), strides=1,
                    padding='same', activation="relu",  # pad default "valid"
                    data_format="channels_last",
                    kernel_initializer='he_normal',
                )
            )
            if batchnorm_every:
                self.convs.append(kl.BatchNormalization())
        if batchnorm_last and not batchnorm_every:
            self.convs.append(kl.BatchNormalization())

        self.downsample = kl.MaxPooling2D(
            pool_size=(2,2), strides=None,
            padding="same", data_format=None,
        )

    def call(self, x):

        for conv in self.convs:
            x = conv(x)
        save = x
        x = self.downsample(x)

        return x, save


class UNet2D(tf.keras.Model):
 
    def __init__(
        self, input_shape, n_classes,
        encoding=None, decoding=None
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

        self.encoders = []
        self.decoders = []

        for nm, argdict in encoding.items():
            self.encoders.append(
                EncodeBlock(**argdict, name=nm)
            )

        # BOTTOM LAYER
        # Consider batchnorms, FC?
        # Make configurable
        self.bottom_block = [
            kl.Conv2D(  # Encode at bottom layer
                64, (2, 2), strides=1, padding="same",
                activation="relu",
                kernel_initializer='he_normal',
                name=f"bottom_{n}"
            ) 
            for n in range(1)
        ]

        # Consider flatten and fully connected here
        for nm, argdict in decoding.items():
            self.decoders.append(
                DecodeBlock(**argdict, name=nm)
            )

        # OUT
        self.out_layer = kl.Conv2D(  # outputs
            n_classes, kernel_size=1, strides=1,
            padding="same", name="last",
        )

        # TODO consider fc layer out?

    def call(self, x):
        # tf.print("IN", x[0])
        encodeds = deque(maxlen=len(self.encoders))
        for encoder in self.encoders:
            x, encoded = encoder(x)
            encodeds.append(encoded)

        # tf.print("encoded", x[0])

        for action in self.bottom_block:
            x = action(x)
        # tf.print("Bottom", x[0])

        for decoder in self.decoders:
            x = decoder(x, encodeds.pop())

        # tf.print("decoded", x[0])

        assert not encodeds

        return self.out_layer(x)
