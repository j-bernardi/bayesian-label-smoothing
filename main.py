import os
import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as kl
import tensorflow.keras.callbacks as cbks


tf.keras.backend.set_floatx('float32')

# TODO - generators for data so memory not exceeded

# TODO neaten conv blocks?
# TODO reduce width more, go into channels?
# Batchnorm on way up?


# TODO - fix up the input shapes. Bring back crop_to?
class UNet2D(tf.keras.Model):
 
    def __init__(self, input_shape, n_classes):
        # height, width, channels

        super(UNet2D, self).__init__()
        w_init = tf.random_normal_initializer()

        self.encode_1 = kl.Conv2D(  # encoder at level 1
            filters=16, kernel_size=(2, 2), strides=1,  # (1, 1), 
            padding='same', activation=None,  # padding default valid
            input_shape=input_shape, data_format="channels_last",
        )
        self.batch_norm_1 = kl.BatchNormalization()
        self.relu_1 = kl.ReLU()
        self.downsample_12 = kl.MaxPooling2D(  # downsample to level 2
            pool_size=(2,2), strides=None, padding="same", data_format=None
        )

        self.encode_2 = kl.Conv2D(32, (2, 2), strides=1, padding="same")
        self.batch_norm_2 = kl.BatchNormalization()
        self.relu_2 = kl.ReLU()
        self.downsample_23 = kl.MaxPooling2D(pool_size=(2,2), padding="same")

        self.level_3 = kl.Conv2D(32, (2, 2), strides=1, padding="same")  # Encode at bottom layer
        self.batch_norm_3 = kl.BatchNormalization()
        self.relu_3 = kl.ReLU()

        # Consider flatten and fully connected here
        
        self.upsample_32 = kl.UpSampling2D(  # upsample level 3 to 2
            size=(2, 2), data_format=None, interpolation='nearest'
        )
        self.concat_last_2 = kl.Concatenate(axis=-1)
        self.decode_2 = kl.Conv2DTranspose(
            16, kernel_size=(2, 2), strides=1, padding="same",
        )

        self.upsample_21 = kl.UpSampling2D(
            size=(2, 2), interpolation='nearest',
            data_format="channels_last",
        )
        self.concat_last_1 = kl.Concatenate(axis=-1)
        self.decode_1 = kl.Conv2DTranspose(  # flexible on size
            16, kernel_size=(2, 2), strides=1, padding="same",
        )

        self.out_layer = kl.Conv2D(  # outputs
            n_classes, kernel_size=1, padding="same"  # TODO or same?
        )

        # TODO consider fc layer out?

    def call(self, x):
        # print("x", x.shape)
        encoded_1 = self.relu_1(
            self.batch_norm_1(
                self.encode_1(x)
            )
        )
        # print("encoded_1", encoded_1.shape)
        level_2_down_input = self.downsample_12(encoded_1)
        # print("l2 down shape", level_2_down_input.shape)
        
        encoded_2 = self.relu_2(
            self.batch_norm_2(
                self.encode_2(level_2_down_input)
            )
        )
        del level_2_down_input
        # print("encoded_2", encoded_2.shape)
        level_3_input = self.downsample_23(encoded_2)
        # print("l3 in", level_3_input.shape)

        level_3_output = self.relu_3(
            self.batch_norm_3(
                self.level_3(level_3_input)
            )
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
        del decoded_2
        level_1_up_input = self.concat_last_1(  # or Concatenate(axis=-1)((x, y))
            (upsampled_from_2, encoded_1)
        )
        del upsampled_from_2, encoded_1
        decoded_1 = self.decode_1(level_1_up_input)
        del level_1_up_input

        return self.out_layer(decoded_1)


def load_data(
        loc="/media/sf_Ubuntu_Shared/datasets/multi_digit_mnist/",
        combined_nm="combined.npy",
        segmented_nm="segmented.npy",
        number=-1,
):

    full_xs = np.load(os.path.join(loc, combined_nm)).astype(np.float32)
    full_ys = np.load(os.path.join(loc, segmented_nm)).astype(np.int32)

    if number > 0:
        xs = full_xs[:number]
        ys = full_ys[:number]
        del full_xs, full_ys
    else:
        xs = full_xs
        ys = full_ys

    print("LOADED DATA", xs.shape)
    print("LOADED LABELS", ys.shape)
    return xs, ys


def make_generator(
        train_xs, train_ys, train_split, val_split,
        train_batch_size, val_batch_size,
):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=(val_split / train_split),
        # Defaults
        featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False, zca_epsilon=1e-06, rotation_range=0, width_shift_range=0.0,
        height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
        channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
        vertical_flip=False, preprocessing_function=None,
        data_format=None, dtype=None
    )
    training_generator = generator.flow(
        x=train_xs, y=train_ys, batch_size=train_batch_size,
        subset="training", shuffle=False,
        # Defaults
        sample_weight=None, seed=None,
        save_to_dir=None, save_prefix='',
        save_format='png',
    )
    validation_generator = generator.flow(
        x=train_xs, y=train_ys, batch_size=val_batch_size,
        subset="validation", shuffle=False, 
        # Defaults
        sample_weight=None, seed=None,
        save_to_dir=None, save_prefix='',
        save_format='png',
    )
    return training_generator, validation_generator


def split_data(dset, split=(0.6, 0.2, 0.2), shuffle=True):

    if shuffle:
        np.random.shuffle(dset)
    assert sum(split) == 1

    lens = [math.floor(s * dset.shape[0]) for s in split]
    lens[-1] = lens[-1] + (dset.shape[0] - sum(lens))
    
    sets = []
    start_from = 0
    for set_len in lens:
        sets.append(dset[start_from:start_from+set_len])
        start_from += set_len

    print(
        "Split into sets", [s.shape[0] for s in sets],
        "from", dset.shape[0]
    )
    return sets


def define_callbacks(
        es_delta=0.0001, es_patience=10,
        rlr_factor=0.33, rlr_patience=5, rlr_delta=0.001, rlr_min=0.001,
):

    early_stop = cbks.EarlyStopping(
        monitor='val_loss', min_delta=es_delta, patience=es_patience,
        restore_best_weights=True, verbose=1, 
        # Defaults
        mode='auto', baseline=None
    )

    reduce_plateau = cbks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.33, patience=5,
        min_delta=0.001, min_lr=0.001,
        verbose=1,
        # Defaults
        cooldown=0, mode='auto',
    )
    tensorboard = cbks.TensorBoard(
        log_dir='tb_logs',
        # Defaults
        # Apparently can't do histograms with generators
        histogram_freq=0, write_graph=True, write_images=False,
        update_freq='epoch', profile_batch=2, embeddings_freq=0,
        embeddings_metadata=None
    )

    return [early_stop, reduce_plateau, tensorboard]


if __name__ == "__main__":
    
    # CONFIGURE
    weights_file = "trained_weights2.h5"
    train_batch_size = val_batch_size = 64
    trn_split, val_split = 0.8, 0.2  # taken out of training only
    optim = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
    )
    loss = tf.keras.losses.CategoricalCrossentropy(
         label_smoothing=0.1  # add custom here
    )
    callbacks = define_callbacks()

    # DATA
    xs, ys = load_data(
        "/export/home/jamesb/Downloads/kaggle/"
        #   "sample_data", "combined_sample.npy", "segmented_sample.npy"
        # number=200,
    )
    
    xs = np.expand_dims(xs, axis=-1).astype(np.float32)
    input_shape = (None, *xs.shape[1:])
    height, width = xs.shape[1:3]
    n_classes = ys.shape[-1]
    print("Input data shape", input_shape)

    train_xs, test_xs = split_data(xs, (trn_split, 1.-trn_split))
    train_ys, test_ys = split_data(ys, (trn_split, 1.-trn_split))
    del xs, ys

    if not os.path.exists(weights_file):
        print("Making dataset generator")
        training_generator, validation_generator = make_generator(
            train_xs, train_ys, trn_split, val_split,
            train_batch_size, val_batch_size,
        )
        train_samples = len(train_xs)
        del train_xs, train_ys

    print("Making model")
    model = UNet2D(input_shape[1:], n_classes)
    model.build(input_shape=input_shape)
    model.summary()
    
    # TRAIN
    if os.path.exists(weights_file):
        model.load_weights(weights_file)
    else:
        model.compile(
            optimizer=optim,
            loss=loss,
            # Defaults
            metrics=None, loss_weights=None,
            weighted_metrics=None, run_eagerly=None,
        )

        model.fit(
            x=training_generator,
            validation_data=validation_generator,
            epochs=500,
            callbacks=callbacks,
            shuffle=False,  # already shuffled
            # Defaults. Ignore steps and batches;
            # generators handle more cleanly
            # (and with repeat data)
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )
        model.save_weights(weights_file)

    # TEST
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in zip(test_xs[:2], test_ys[:2]):
        logits = model(np.expand_dims(x, axis=0) / 255)  # rescale
        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        flat_y = tf.argmax(np.expand_dims(y, axis=0), axis=-1)
        test_accuracy(prediction, flat_y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))
