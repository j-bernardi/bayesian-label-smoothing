import numpy as np
import tensorflow as tf

from utils import adjacency_from_generator, GLOBAL_TYPE

tf.keras.backend.set_floatx(GLOBAL_TYPE)

## NOTE TO SELF (and to document): 
##   All self-adjacencies are doubled
##   in terms of perimeter:border ratio
##   BUT bear in mind it's prop. to
##   number of pixels on a border vs in an area


def fixed_uniform_smoothing(
    n_classes,
    fixed_smoothing_magnitude=0.9,
):
    """Smoothing matrix for fixed-uniform regime

    Just the same as traditional label smoothing, Included for
    consistency
    """
    smoothing_matrix = np.zeros((n_classes, n_classes))
    for c in range(n_classes):
        smooth = np.zeros(n_classes) + (
            1. - fixed_smoothing_magnitude) / (n_classes - 1)
        smooth[c] = fixed_smoothing_magnitude
        assert np.sum(smooth) == 1.
        smoothing_matrix[c] = smooth
    return smoothing_matrix


def fixed_adjacent_smoothing(
    n_classes,
    training_generator,
    fixed_smoothing_magnitude=0.9,
    generator_length=-1,
):
    """Smoothing matrix for fixed-adjecent regime

    Return the label smoothing matrix for fixed magnitude
    smoothing, with other classes' smoothing defined by
    the adjacency of each label-class to their neighbours
    """
    # Fractional weights per class
    adjacency = adjacency_from_generator(
        n_classes, training_generator, generator_length
    )
    smoothing_matrix = np.zeros((n_classes, n_classes))

    for c in range(n_classes):
        remainder_adj_row = np.array(adjacency[c], copy=True)
        # Ignore self-adjacency. Magnitude doesn't affect.
        # Normalise off-diagonal
        remainder_adj_row[c] = 0.
        weights = remainder_adj_row / remainder_adj_row.sum()
        # Fill smoothing into weighted amounts
        smoothing_matrix[c] =\
            np.ones(n_classes) * weights * (
                1. - fixed_smoothing_magnitude
            )
        smoothing_matrix[c, c] = fixed_smoothing_magnitude
        assert np.sum(smoothing_matrix[c]) == 1.
    return smoothing_matrix


def weighted_uniform_smoothing(
    n_classes,
    training_generator,
    max_smoothing_magnitude=0.8,
    generator_length=-1,
):
    """Smoothing matrix for weighted-uniform regime
    
    Return the label smoothing matrix for variable
    magnitude smoothing, where magnitude is defined by:
        max_mag + (self-adj / (sum of others)) * (1-max_mag)
    E.g.how adjacent is is to other classes vs itself
    (inverese prop. to perim:volume) with other classes,
    smoothed uniformly to other classes.
    """

    smoothing_matrix = np.zeros((n_classes, n_classes))
    adjacency = adjacency_from_generator(
        n_classes, training_generator, generator_length
    )
    for c in range(n_classes):
        self_adj_ratio = adjacency[c, c] / adjacency[c].sum()
        label_adjustment = self_adj_ratio * (
            1. - max_smoothing_magnitude
        )
        new_label_mag = max_smoothing_magnitude + label_adjustment
        epsilon = (1. - new_label_mag) / (n_classes - 1)
        smoothing_matrix[c, :] = epsilon
        smoothing_matrix[c, c] = new_label_mag
        assert smoothing_matrix[c].sum() == 1.
    return smoothing_matrix


def weighted_adjacent_smoothing(
    n_classes,
    training_generator,
    max_smoothing_magnitude=0.8,
    generator_length=-1,
):
    """Smoothing matrix for weighted-adjecent regime
    
    Return the label smoothing matrix for variable
    magnitude smoothing, where magnitude is defined by:
        max_mag + (self-adj / (sum of others)) * (1-max_mag)
    E.g.how adjacent is is to other classes vs itself
    (inverese prop. to perim:volume) with other classes,
    smoothed weighted by adjacency to other classes.
    """
    smoothing_matrix = np.zeros((n_classes, n_classes))
    adjacency = adjacency_from_generator(
        n_classes, training_generator, generator_length
    )
    for c in range(n_classes):
        self_adj_ratio = adjacency[c, c] / adjacency[c].sum()
        label_adjustment = self_adj_ratio * (
            1. - max_smoothing_magnitude
        )
        new_label_mag = max_smoothing_magnitude + label_adjustment
        # Weight how the remainder is shared:
        #   Ignore self-adjacency. Magnitude doesn't affect 
        #   weighting. Then normalise the off-diagonal
        remainder_adj_row = np.array(adjacency[c], copy=True)
        remainder_adj_row[c] = 0.
        weights = remainder_adj_row / remainder_adj_row.sum()

        smoothing_matrix[c] =\
            np.ones(n_classes) * weights * (1. - new_label_mag)
        smoothing_matrix[c, c] = new_label_mag
        assert smoothing_matrix[c].sum() == 1.
    return smoothing_matrix
