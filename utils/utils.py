import numpy as np
import tensorflow as tf

from .globals import GLOBAL_TYPE

tf.keras.backend.set_floatx(GLOBAL_TYPE)


def class_sums_from_generator(n_classes, training_generator, generator_length=-1):
    """Return an array of class counts in a dataset

    training_generator: when iterated for num_training_batches
        (or completion, if non-repeating), returns the whole 
        set in batches of (x, labs)
    generator_length: break the generator iteration, if generator
        repeats (e.g. is infinite)

    Returns:
        class_sums (np.array): integer sums of each class present
            in the generator
    """
    class_sums = np.zeros((n_classes,), dtype=np.int64)
    for b, (_, label) in enumerate(training_generator):
        classes, _,  counts = tf.unique_with_counts(
            tf.reshape(tf.argmax(label, axis=-1), [-1])
        )
        for i in range(len(classes)):
            class_sums[classes[i]] += counts[i]
        if b == generator_length:
            break
    return class_sums

# NOTE: perhaps instead of for loop we could be gathering
# E.g. adjacency[label, class(rolled)] = counts(rolled)
def adjacency_from_generator(n_classes, training_generator, generator_length=-1):
    """Calculate class adjencies of dataset

    generator dim 4 req, (batch, x, y, class)

    Returns:
        adjacency: np.arrary shape (n_classes, n_classes)
            A matrix of adjacencies, per class

    """

    # TODO axis and shift aren't generalisable: 1 or -1
    def update_adjacency(
        adj_matrix, adj_to_class, 
        flattened_labels,
        axis, shift
    ):
        """
        Count everything adjacent (defined by axis, shift)
        to instances of the class "adj_to_class"
        in a given batch of flattened_labels

        Add to the adj_matrix[adj_to_class] += counts
        """
        def crop_out(arr, shift, axis):
            """Crop out the edge of an arr

            Crop an edge in a given dimension by slicing array
            from that axis. Has effect of shifting elements by
            a given shift in given axis, with no roll-around.
            """
            if len(arr.shape) != 3:
                raise NotImplementedError(
                    f"Dim must be 3 not {arr.shape}"
                )
            if axis == 1:
                if shift > 0:
                    cropped = arr[:, :-shift, :]
                else:
                    cropped = arr[:, -shift:, :]
            elif axis == 2:
                if shift > 0:
                    cropped = arr[:, :, :-shift]
                else:
                    cropped = arr[:, :, -shift:]
            return cropped
        adjacent_classes = tf.where(
            # Redefine labels having removed axis that will
            # be rolled-around by the adjacency-finding shift
            crop_out(flattened_labels, -shift, axis) == adj_to_class,
            # Shift labels to put adjacencies in-place of the
            # original label cell
            crop_out(flattened_labels, shift, axis),
            # Blank-out the cells that we are not counting from
            # (e.g. that are not adjacent to adj_to_class)
            -1,
        )
        classes, _, counts = tf.unique_with_counts(
            tf.reshape(adjacent_classes, [-1])
        )
        # TODO is this a gather?
        for i in range(len(classes)):
            # -1 used to indicate non-adjacent cells
            if classes[i] >= 0:
                adj_matrix[c, classes[i]] += counts[i]

    adjacency = np.zeros((n_classes, n_classes))

    for b, (_, label) in enumerate(training_generator):
        assert len(label.shape) == 4
        flat_label = tf.argmax(label, axis=-1)
        for c in range(n_classes):
            for counting, ax, shf in (
                ("above", 1, 1), ("below", 1, -1),
                ("left", 2, 1), ("right", 2, -1),
            ):
                update_adjacency(
                    adjacency, c,
                    flat_label,
                    ax, shf,
                )

        if b == generator_length:
            break

    return adjacency
