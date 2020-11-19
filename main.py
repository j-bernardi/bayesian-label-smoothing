import os
import sys
import math
import datetime
import shutil
import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as cbks

from models.unet_2d import UNet2D
from models.mobile_net_example import UNetExample

tf.keras.backend.set_floatx('float32')


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
        es_delta=0.0001, es_patience=8,
        rlr_factor=0.33, rlr_patience=4, 
        rlr_delta=0.001, rlr_min=0.00001,
):

    early_stop = cbks.EarlyStopping(
        monitor='val_loss', min_delta=es_delta,
        patience=es_patience, verbose=1,
        restore_best_weights=True,
        # Defaults
        mode='auto', baseline=None
    )

    reduce_plateau = cbks.ReduceLROnPlateau(
        monitor='val_loss', factor=rlr_factor,
        patience=rlr_patience, min_delta=rlr_delta, 
        min_lr=rlr_min, verbose=1,
        # Defaults
        cooldown=0, mode='auto',
    )
    
    log_dir = "tb_logs/" +\
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = cbks.TensorBoard(
        log_dir=log_dir, update_freq='epoch',  # profile_batch=0,
        histogram_freq=1,
        # Defaults
        # Bug reported elsewhere: can't do histograms with generators
        write_graph=True, write_images=False,
        embeddings_freq=0, embeddings_metadata=None
    )

    return [early_stop, reduce_plateau, tensorboard]


if __name__ == "__main__":

    exp_dir = os.path.join("experiments", sys.argv[1])
    os.makedirs(exp_dir, exist_ok=True)
    weights_file = os.path.join(exp_dir, "weights.h5")
    pretrained = os.path.exists(weights_file)
    print("Weights", weights_file, "pretrained:", pretrained)

    # CONFIGURE
    config_file = os.path.join(exp_dir, "config.py")
    if not os.path.exists(config_file):
        shutil.copy("template_config.py", config_file)
    exec(open(config_file).read(), globals(), locals())

    if not pretrained:
        callbacks = define_callbacks(**callback_args)

    # SELECT DATA
    xs, ys = load_data(
        "/export/home/jamesb/Downloads/kaggle/"
        #   "sample_data", "combined_sample.npy", "segmented_sample.npy"
        # number=200,
    )

    # Mobilenet (to RGB)
    if mobile_net:
        xs = np.repeat(xs[..., np.newaxis], 3, -1).astype(np.float32)
        xs = tf.image.resize(
            xs, (96, 96),
            # DEFAULTS
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False, antialias=False, name=None
        )
        ys = tf.image.resize(
            ys, (96, 96),
            # DEFAULTS
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, 
            preserve_aspect_ratio=False, antialias=False, name=None
        )
    elif unet:
        # Indicate grayscale
        xs = np.expand_dims(xs, axis=-1).astype(np.float32)
    ###
    
    input_shape = (None, *xs.shape[1:])
    n_classes = ys.shape[-1]
    print("Input data shape", input_shape)
    train_xs, test_xs = split_data(xs, (trn_split, 1.-trn_split), shuffle=False)
    train_ys, test_ys = split_data(ys, (trn_split, 1.-trn_split), shuffle=False)
    del xs, ys

    if not pretrained:
        print("Making dataset generator")
        training_generator, validation_generator = make_generator(
            train_xs, train_ys, trn_split, val_split,
            train_batch_size, val_batch_size,
        )
        train_samples = len(train_xs)
        del train_xs, train_ys

    print("Making model")
    if mobile_net:
        model = UNetExample(input_shape[1:], n_classes)
    elif unet:
        model = UNet2D(
            input_shape[1:], n_classes, 
            encoding, decoding
        )
    model.build(input_shape=input_shape)
    model.summary()

    # TRAIN
    if pretrained:
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
            epochs=max_epochs,
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

    # TEST. TODO - get a per-class accuracy?
    test_accuracy = tf.keras.metrics.Accuracy()

    for (x, y) in zip(test_xs, test_ys):
        logits = model(np.expand_dims(x, axis=0) / 255)  # rescale
        prediction = tf.argmax(logits, axis=-1, output_type=tf.int32)
        flat_y = tf.argmax(np.expand_dims(y, axis=0), axis=-1)
        test_accuracy(prediction, flat_y)

    result_string = "Test set accuracy: {:.3%}".format(
        test_accuracy.result()
    )
    print(result_string)
    with open(os.path.join(exp_dir, "results.txt"), "w") as rf:
        rf.write(result_string)
