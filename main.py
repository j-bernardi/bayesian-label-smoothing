import os
import sys
import csv
import math
import datetime
import shutil
import pickle
import argparse
import numpy as np
import tensorflow as tf
import tensorflow.keras.callbacks as cbks
import matplotlib.pyplot as plt

from models.unet_2d import UNet2D
from utils import GLOBAL_TYPE, class_sums_from_generator, Config
from losses import (
    CustomSmoothedWeightedCCE,
    fixed_uniform_smoothing,
    fixed_adjacent_smoothing,
    weighted_uniform_smoothing,
    weighted_adjacent_smoothing,
)

tf.keras.backend.set_floatx(GLOBAL_TYPE)


def load_data(
        loc="data",
        combined_nm="combined.npy",
        segmented_nm="segmented.npy",
        number=-1,
):

    if number == "sample":
        loc = "sample_data"
        combined_nm = "combined_sample.npy"
        segmented_nm = "segmented_sample.npy"
        xs = np.load(os.path.join(loc, combined_nm)).astype(np.float32)
        ys = np.load(os.path.join(loc, segmented_nm)).astype(np.int32)
    else:
        full_xs = np.load(os.path.join(loc, combined_nm)).astype(np.float32)
        full_ys = np.load(os.path.join(loc, segmented_nm)).astype(np.int32)

        if number > 0:
            xs = full_xs[:number]
            ys = full_ys[:number]
            del full_xs, full_ys
        elif number < 0:
            xs = full_xs
            ys = full_ys
        else:
            raise ValueError(
                f"Cannot run with {number} data, try \"sample\""
            )

    n_classes = ys.shape[-1]
    print("LOADED DATA", xs.shape)
    print("LOADED LABELS", ys.shape)
    print("Classes", n_classes)
    return xs, ys, n_classes


def make_generator(
        train_xs,
        train_ys,
        train_split,
        val_split,
        train_batch_size,
        val_batch_size,
):
    generator = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=(val_split / train_split),
        rotation_range=10.0,
        width_shift_range=2.,
        height_shift_range=2.,
        shear_range=0.0,
        zoom_range=0.1,
        # Defaults
        featurewise_center=False, samplewise_center=False,
        featurewise_std_normalization=False, samplewise_std_normalization=False,
        zca_whitening=False, zca_epsilon=1e-06,
        brightness_range=None,
        channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
        vertical_flip=False, preprocessing_function=None,
        data_format=None, dtype=None,
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


def get_class_weights(
    n_classes, training_generator,
    mode="uniform",
    generator_length=-1,
    background_idx=10,
    drop_background=0.10
):
    """Iterate training data to get balancing weights

    Args:
        n_classes: total number of classes
        training_generator: when iterated for
            num_training_batches (or completion, if non-repeating),
            returns the whole set in batches of (x, labs)
        generator_length: break the generator iteration, if
            generator repeats (e.g. is infinite).
        mode: Select mode from:
            uniform: 1s
            drop_background: set background_idx to 0.05 (else 1.)
            balance_off_max: max value will be set to 1. Rest will be
                max(a) / a for each value a (count of labels)
            balance_off_min: min value will be set to 1. Rest will be
                1 / a for each value a (count of labels)
            balance_off_median: min value will be set to 1. Rest will be
                1 / a for each value a (count of labels)
    """
    if mode == "uniform":
        class_weights = {c: 1. for c in range(n_classes)}

    elif mode == "drop_background":
        # TEMP:  hardcoded 5%
        class_weight_list = [1. for _ in range(n_classes)]
        class_weight_list[background_idx] = drop_background
        class_weights = {
            c: class_weight_list[c] for c in range(n_classes)
        }
    else:
        class_sums = class_sums_from_generator(
            n_classes, training_generator, generator_length
        )
        balanced_weights = [
            np.sum(class_sums) / class_sums[i]
            for i in range(n_classes)
        ]
    if mode == "balance_off_max":
        # "Max" class (background) has weight 1 so no divis.
        class_weights = {
            i: balanced_weights[i] for i in range(n_classes)
        }
    elif mode == "balance_off_min":
        class_weights = {
            i: balanced_weights[i] / np.max(balanced_weights)
            for i in range(n_classes)
        }
    elif mode == "balance_off_med":
        class_weights = {
            i: balanced_weights[i] / np.median(balanced_weights)
            for i in range(n_classes)
        }
    return class_weights


def define_callbacks(
    exp_dir,
    es_delta=0.0001, es_patience=8,
    rlr_factor=0.33, rlr_patience=4, 
    rlr_delta=0.001, rlr_min=0.00001,
):

    early_stop = cbks.EarlyStopping(
        monitor='val_loss', min_delta=es_delta,
        patience=es_patience, verbose=2,
        restore_best_weights=True,
        # Defaults
        mode='auto', baseline=None
    )

    reduce_plateau = cbks.ReduceLROnPlateau(
        monitor='val_loss', factor=rlr_factor,
        patience=rlr_patience, min_delta=rlr_delta, 
        min_lr=rlr_min, verbose=2,
        # Defaults
        cooldown=0, mode='auto',
    )    
    log_dir = (
        os.path.join(
            "tb_logs",
            f"{exp_dir}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
    )
    tensorboard = cbks.TensorBoard(
        log_dir=log_dir, update_freq='epoch',  # profile_batch=0,
        histogram_freq=1,
        # Defaults
        # Bug reported elsewhere: can't do histograms with generators
        write_graph=True, write_images=False,
        embeddings_freq=0, embeddings_metadata=None
    )

    return [early_stop, reduce_plateau, tensorboard]


def display_multi(xs, ys, args):
    """Display random n images, save to exp dir

    """
    plt.axis('off')
    fig, axes = plt.subplots(nrows=args.display, ncols=3, figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    rand_idxs = np.random.choice(len(xs), size=args.display, replace=False)
    for n, (rand_idx, plot_row) in enumerate(zip(rand_idxs, axes)):
        # gather data
        rand_x = xs[rand_idx]
        rand_y = tf.argmax(ys[rand_idx], axis=-1)
        rand_pred = tf.argmax(
            model(np.expand_dims(rand_x, axis=0) / 255),
            axis=-1
        )[0]
        display_list = [rand_x, rand_y, rand_pred]
        
        # Plot subplots
        for i, plot_col in enumerate(plot_row):
            if n == 0:
                plot_col.set_title(title[i])
            try:
                disp = tf.keras.preprocessing.image.array_to_img(
                    display_list[i]
                )
            except:
                disp = display_list[i]
            plot_col.imshow(disp)
    
    # Save and show
    fig_file = os.path.join(args.exp_dir, "example_segment.png")
    if overwrite(fig_file, args):
        fig.savefig(fig_file)
    plt.axis('on')  # reset


def parse_my_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "exp_dir", type=str,
        help="A path to an experiment directory, optionally including a "
        "config.py file which contains a variable called config = Config(). "
        "See defaults/baseline_config.py for example."
    )
    parser.add_argument(
        "--data-num", "-n", default=-1,
        help="Number of datapoints to train on, integer or \"sample\""
             "to use sample data in repo"
    )
    parser.add_argument(
        "--display", "-d", default=0, type=int,
        help="Display n example images of the network at the end with pyplot"
    )
    parser.add_argument("--force", "-f", action="store_true",
        help="Overwrites current weights and graphs"
    )
    args = parser.parse_args()

    if args.data_num != "sample":
        args.data_num = int(args.data_num)

    return args


def overwrite(filename, args):
    """Whether to overwrite a file given arg config

    Written out for clear logic
    """
    file_exists = os.path.exists(filename)
    sample_run = args.data_num == "sample"

    if sample_run or (file_exists and not args.force):
        return False
    else:
        return True


if __name__ == "__main__":
    args = parse_my_args()
    result_file = "results.csv"

    # FILES
    os.makedirs(args.exp_dir, exist_ok=True)
    weights_file = os.path.join(args.exp_dir, "weights.h5")
    history_file = os.path.join(args.exp_dir, "history.p")
    train = not os.path.exists(weights_file) or args.force
    print("Weights", weights_file, "training? :", train)

    # CONFIGURE
    config_file = os.path.join(args.exp_dir, "config.py")
    if not os.path.exists(config_file):
        print("No config, using default")
        config = Config()  # defaults
    else:
        exec(open(config_file).read(), globals(), locals())

    # Fail early if invalid
    if train:
        callbacks = define_callbacks(args.exp_dir, **config.callback_args)

    # SELECT DATA
    xs, ys, n_classes = load_data(number=args.data_num)
    xs = np.expand_dims(xs, axis=-1).astype(np.float32)  # Indicate grayscale
    input_shape = (None, *xs.shape[1:])
    print("Input data shape", input_shape)
    split = (config.train_split, 1.-config.train_split)
    train_xs, test_xs = split_data(xs, split, shuffle=False)
    train_ys, test_ys = split_data(ys, split, shuffle=False)
    num_training_batches = (
        len(train_xs) * config.train_split // config.train_batch_size
    )
    del xs, ys

    # MAKE MODEL
    print("Making model")
    model = UNet2D(
        input_shape[1:], n_classes, 
        encoding=config.encoding,
        decoding=config.decoding,
        central=config.central,
    )
    model.build(input_shape=input_shape)
    model_summary = model.summary()

    # Skip slow step if pretrained
    if train:
        print("Making dataset generator")
        training_generator, validation_generator = make_generator(
            train_xs,
            train_ys,
            config.train_split,
            config.val_split,
            config.train_batch_size,
            config.val_batch_size,
        )
    train_samples = len(train_xs)
    del train_xs, train_ys

    # TRAIN (or load)
    if not train:
        model.load_weights(weights_file)
        history = None
        if os.path.exists(history_file):
            with open(history_file, "rb") as f:
                history = pickle.load(f)
            print("Loaded history", history_file)
        else:
            print("Could not find history file", history_file)
    else:
        print(f"Getting class weights {config.class_weight_mode}...")
        class_weights = get_class_weights(
            n_classes,
            training_generator,
            mode=config.class_weight_mode,
            generator_length=num_training_batches,
            drop_background=config.drop_background,
        )
        print("Class weights calculated", class_weights)

        print(f"Getting smoothing matrix {config.smoothing_function.__name__}")
        if config.smoothing_function is None:
            smoothing_matrix = None
        else:
            smoothing_matrix = config.smoothing_function(
                n_classes, training_generator, num_training_batches
            )
        weighted_cce = CustomSmoothedWeightedCCE(
            class_weights=list(class_weights.values()),
            label_smoothing=smoothing_matrix,
            from_logits=True,
            **config.loss_args
        )
        model.compile(
            optimizer=config.optim,
            loss=weighted_cce,
            metrics=['accuracy'],
            # Defaults to consider
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
        )
        train_report = model.fit(
            x=training_generator,
            validation_data=validation_generator,
            epochs=config.max_epochs,
            callbacks=callbacks,
            shuffle=False,  # already shuffled
            verbose=1,
            # Defaults. Ignore steps and batches;
            # generators handle more cleanly
            # (and with repeat data)
            sample_weight=None,
            class_weight=None,  # handled with customLF
            initial_epoch=0,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False
        )
        history = train_report.history
        if overwrite(weights_file, args):
            model.save_weights(weights_file)
        if overwrite(history_file, args):
            with open(history_file, "wb") as f:
                pickle.dump(history, f, protocol=pickle.HIGHEST_PROTOCOL)

    # LOG TRAINING
    loss_graph_file = os.path.join(args.exp_dir, "losses.png")
    if history is not None and overwrite(loss_graph_file, args):
        plt.figure()
        epochs = list(range(len(history["loss"])))
        plt.plot(epochs, history["loss"], label="Training Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.title("Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(loss_graph_file)
    else:
        print("Loss graph not (re)saved")

    # TEST. TODO - batch
    print("Evaluating...")
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for (x, y) in zip(test_xs, test_ys):
        logits = model(np.expand_dims(x, axis=0) / 255)  # rescale
        prediction = tf.argmax(logits, axis=-1, output_type=tf.int64)
        flat_y = tf.argmax(
            np.expand_dims(y, axis=0), axis=-1, output_type=tf.int64
        )
        img_confusion = tf.math.confusion_matrix(
            labels=tf.reshape(flat_y, [-1]),
            predictions=tf.reshape(prediction, [-1]),
            num_classes=n_classes, dtype=tf.int64,
        ).numpy()
        cm += img_confusion
    print("Complete")

    # Process CM and save raw
    cm_file = os.path.join(args.exp_dir, "confusion.csv")
    if overwrite(cm_file, args):
        np.savetxt(cm_file, cm, delimiter=",")
    cm = cm.astype('double')  # Cast, for calculations
    total_accuracy = np.sum(np.diagonal(cm)) / np.sum(cm)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    accuracy_per_class = np.diagonal(cm)  # TP accuracy
    avg_accuracy_per_class = np.mean(accuracy_per_class)
    avg_accuracy_per_target_class = np.mean(accuracy_per_class[:-1])

    # Create report
    result_string = "Test set accuracy: {:.3%}".format(total_accuracy)
    if history is not None:
        result_string += "\n\nFinal val loss: {:.3f}".format(history["val_loss"][-1])
    result_string += "\n\nAvg accuracy per class: {:.3%}".format(
        avg_accuracy_per_class
    )
    result_string += "\n\nAvg accuracy per class exc background: {:.3%}".format(
        avg_accuracy_per_target_class
    )
    #  Beautify the output cm
    np.set_printoptions(precision=3, suppress=True)
    result_string += "\n\nConfusion:\n" + str(cm)

    result_string += f"\n\n{model_summary}"

    # Write to file and display
    result_file = os.path.join(args.exp_dir, "results.txt")
    if overwrite(result_file, args):
        with open(result_file, "w") as rf:
            rf.write(result_string)
    else:
        print("WARNING, force false, did not write results")

    # Write csv to keep track
    # Only track full runs
    if True:  # args.data_num == -1:
        # Add titles
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    "exp", "acc", "avg_cls_acc", "avg_cls_acc_exc_bg",
                    "val_loss"
                ])
        csv_line = [
            args.exp_dir, total_accuracy, avg_accuracy_per_class,
            avg_accuracy_per_target_class, history["val_loss"][-1]
        ]
        # Append result
        with open(result_file, "a") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_line)

    print(result_string)

    # Display (and save in-function) n random images
    if args.display:
        display_multi(test_xs, test_ys, args)
        plt.show()
