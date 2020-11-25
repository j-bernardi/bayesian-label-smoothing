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
        adj_row = np.array(adjacency[c], copy=True)
        # Ignore self-adjacency. Magnitude doesn't affect.
        # Normalise off-diagonal
        adj_row[c] = 0.
        weights = adj_row / adj_row.sum()
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
    magnitude smoothing, where magnitude is defined by
    how adjacent is is to other classes (inverese prop.
    to perim:volume) with other classes, smoothed
    uniformly to other classes.
    """
    raise NotImplementedError("Not yet implemented")
    smoothing_matrix = np.zeros((n_classes, n_classes))
    adjacency = adjacency_from_generator(
        n_classes, training_generator, generator_length
    )
    # TODO - weight magnitude by self-adjacency
    # Smooth uniformly
    return smoothing_matrix

def weighted_adjacent_smoothing(
    n_classes,
    training_generator,
    max_smoothing_magnitude=0.8,
    generator_length=-1,
):
    """Smoothing matrix for weighted-adjecent regime
    Comment
    """
    raise NotImplementedError("Not yet implemented")
    smoothing_matrix = np.zeros((n_classes, n_classes))
    adjacency = adjacency_from_generator(
        n_classes, training_generator, generator_length
    )
    # TODO - weight magnitude and get adjacency
    return smoothing_matrix
