import tensorflow as tf
import tensorflow.keras.layers as kl
from collections import deque


class DecodeBlock(tf.keras.layers.Layer):
    """Layer to decode a UNet

    """

    def __init__(
        self, num_filters_out, num_convs=1, name="decode",
        batchnorm_every=False, batchnorm_last=False,
        crop_to_shape=None,
    ):
        """
        Args:
            num_filters_out: number of filters in
                the layer we're decoding *to*
            crops: a tuple of (target_height, target_width) 
                to crop the upsampled tensor to match the
                (target of) the encoded tensor shape, 
                if difference expected
        """
        super(DecodeBlock, self).__init__(name=name)

        self.upsample = kl.Conv2DTranspose(
            num_filters_out, kernel_size=(2, 2), strides=2, 
            padding='same', activation="relu",
            data_format=None, kernel_initializer='he_normal',
        )
        self.concat = kl.Concatenate(axis=-1)
        self.crop_to_shape = crop_to_shape

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
        if self.crop_to_shape is not None:
            x = tf.image.resize_with_crop_or_pad(
                x, *self.crop_to_shape
            )
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

