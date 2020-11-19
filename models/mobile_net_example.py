import tensorflow as tf
import tensorflow.keras.layers as kl

from tensorflow_examples.models.pix2pix import pix2pix


class UNetExample(tf.keras.Model):

    def __init__(self, input_shape, output_classes, name="unet_example"):

        super(UNetExample, self).__init__(name=name)

        # self.my_input_layer = kl.Input(shape=input_shape)

        base_model = tf.keras.applications.MobileNetV2(
            input_shape=list(input_shape), 
            include_top=False
        )

        # Use the activations of these layers
        layer_names = [
            'block_1_expand_relu',   # 64x64
            'block_3_expand_relu',   # 32x32
            'block_6_expand_relu',   # 16x16
            'block_13_expand_relu',  # 8x8
            'block_16_project',      # 4x4
        ]
        layers = [base_model.get_layer(name).output for name in layer_names]
        
        # Create the feature extraction model
        self.down_stack = tf.keras.Model(
            inputs=base_model.input,
            outputs=layers
        )

        self.down_stack.trainable = False
        
        self.up_stack = [
            pix2pix.upsample(512, 3),  # 4x4 -> 8x8
            pix2pix.upsample(256, 3),  # 8x8 -> 16x16
            pix2pix.upsample(128, 3),  # 16x16 -> 32x32
            pix2pix.upsample(64, 3),   # 32x32 -> 64x64
        ]

        self.concats = [
            tf.keras.layers.Concatenate() 
            for _ in range(len(self.up_stack))
        ]

        self.last = kl.Conv2DTranspose(  #64x64 -> 128x128
            output_classes, 3, strides=2,
            padding='same'
        )

    def call(self, x):

        # Downsampling through the model
        skips = self.down_stack(x)
        x = skips[-1]
        skips = reversed(skips[:-1])

        # Upsampling and establishing the skip connections
        for i, (concat_layer, up_layer, skip) in \
                enumerate(zip(self.concats, self.up_stack, skips)):
            x = up_layer(x)
            x = concat_layer([x, skip])
        
        return self.last(x)
