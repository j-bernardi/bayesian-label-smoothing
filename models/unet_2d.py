import tensorflow as tf
import tensorflow.keras.layers as kl


class UNet2D(tf.keras.Model):
 
    def __init__(self, input_shape, n_classes):
        # height, width, channels

        super(UNet2D, self).__init__()
        w_init = tf.random_normal_initializer()

        ### TODO ###
        # Rename "encode" as it's whole block, really
        ############

        self.encode_1 = kl.Conv2D(  # encoder at level 1
            filters=16, kernel_size=(2, 2), strides=1,  # (1, 1), 
            padding='same', activation="relu",  # padding default valid
            input_shape=input_shape, data_format="channels_last",
        )
        # self.relu_1 = kl.ReLU()
        self.batch_norm_1 = kl.BatchNormalization()
        self.downsample_12 = kl.MaxPooling2D(  # downsample to level 2
            pool_size=(2,2), strides=None, padding="same", data_format=None
        )

        self.encode_2 = kl.Conv2D(
            32, (2, 2), strides=1, padding="same",
            activation="relu",
        )
        # self.relu_2 = kl.ReLU()
        self.batch_norm_2 = kl.BatchNormalization()
        self.downsample_23 = kl.MaxPooling2D(pool_size=(2,2), padding="same")

        self.level_3 = kl.Conv2D(  # Encode at bottom layer
            32, (2, 2), strides=1, padding="same",
            activation="relu",
        )
        # self.relu_3 = kl.ReLU()
        self.batch_norm_3 = kl.BatchNormalization()

        # Consider flatten and fully connected here

        ### TODO ###
        # Should be up-conv to upsample (and get rid of UpSampling2D),
        # then process concatenated blocks with 2x convs.
        # Also, people recommend no batchnorm on 2nd leg,
        # as it's expensive and you're inputting concat BN
        # Make sure to activate the convs.
        ############

        self.upsample_32 = kl.Conv2DTranspose(  # upsample level 3 to 2
            32, kernel_size=(2, 2), strides=2, 
            padding='same', activation="relu",
            data_format=None,
        )
        self.concat_last_2 = kl.Concatenate(axis=-1)
        self.decode_2 = kl.Conv2D(
            32, kernel_size=(2, 2), strides=1, 
            padding="same", activation="relu",
        )

        self.upsample_21 = kl.Conv2DTranspose(  # upsample level 3 to 2
            16, kernel_size=(2, 2), strides=2, 
            padding='same', activation="relu",
            data_format=None,
        )
        self.concat_last_1 = kl.Concatenate(axis=-1)
        self.decode_1 = kl.Conv2D(  # flexible on size
            16, kernel_size=(2, 2), strides=1,
            padding="same", activation="relu",
        )

        self.out_layer = kl.Conv2D(  # outputs
            n_classes, kernel_size=1, strides=1,
            padding="same",
        )

        # TODO consider fc layer out?

    def call(self, x):
        # print("x", x.shape)
        encoded_1 = self.batch_norm_1(
            self.encode_1(x)
        )
        # print("encoded_1", encoded_1.shape)
        level_2_down_input = self.downsample_12(encoded_1)
        # print("l2 down shape", level_2_down_input.shape)
        
        encoded_2 = self.batch_norm_2(
            self.encode_2(level_2_down_input)
        )
        del level_2_down_input
        # print("encoded_2", encoded_2.shape)
        level_3_input = self.downsample_23(encoded_2)
        # print("l3 in", level_3_input.shape)

        level_3_output = self.batch_norm_3(
            self.level_3(level_3_input)
        )
        del level_3_input

        # Consider fully connected in middle
        # print("l3 out", level_3_output.shape)
        upsampled_from_3 = self.upsample_32(level_3_output)
        # print("l3 ups", upsampled_from_3.shape)
        # print("Catting with encoded2", encoded_2.shape)
        level_2_up_input = self.concat_last_2(
            (upsampled_from_3, encoded_2)
        )
        del upsampled_from_3, encoded_2
        # print("catted shape", level_2_up_input.shape)
        decoded_2 = self.decode_2(level_2_up_input)
        # print("decoded shape 2", decoded_2.shape)

        upsampled_from_2 = self.upsample_21(decoded_2)
        # print("Upsampled from 2", upsampled_from_2.shape)
        del decoded_2
        level_1_up_input = self.concat_last_1(  # or Concatenate(axis=-1)((x, y))
            (upsampled_from_2, encoded_1)
        )
        del upsampled_from_2, encoded_1
        decoded_1 = self.decode_1(level_1_up_input)
        del level_1_up_input

        return self.out_layer(decoded_1)
