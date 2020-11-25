import numpy as np

from losses import (
    fixed_uniform_smoothing,
    fixed_adjacent_smoothing,
    weighted_uniform_smoothing,
    weighted_adjacent_smoothing,
)


def test_fixed_uniform_smooth():

    n_classes = 3
    magnitude = 0.95

    smooth = fixed_uniform_smoothing(
        n_classes,
        fixed_smoothing_magnitude=magnitude,
    )

    print(f"Smooth\n{smooth}")

    expected = np.array([
        [0.95, 0.025, 0.025],
        [0.025, 0.95, 0.025],
        [0.025, 0.025, 0.95],
    ])

    assert np.all(np.isclose(smooth, expected))


def test_fixed_adjacent_smooth():

    n_classes = 3
    fixed_smoothing_magnitude = 0.94
    x = np.zeros((1, 3, 3, 3), dtype=int)
    x[0, :, :, 0] = 1  # All class 0
    x[0, 1, 1, 0] = 0  # Switch middle to 1
    x[0, 1, 1, 1] = 1
    x[0, 2, 2, 0] = 0  # Switch LwRt to 1
    x[0, 2, 2, 2] = 1
    # label > adj
    # [0, 0, 0],  12, 4, 2
    # [0, 1, 0],   4, 0, 0
    # [0, 0, 2],   2, 0, 0
    gen = iter([(None, x), ])

    smooth = fixed_adjacent_smoothing(
        n_classes,
        fixed_smoothing_magnitude=fixed_smoothing_magnitude,
        training_generator=gen,
        generator_length=1,
    )
    print(f"Smooth\n{smooth}")

    expected = np.array([
        [0.94, 0.04,  0.02], 
        [0.06, 0.94,    0.],
        [0.06, 0.,    0.94],
    ])


    assert np.all(np.isclose(smooth, expected))


def test_weighted_uniform_smooth():

    n_classes = 3
    max_smoothing_magnitude = 0.8
    x = np.zeros((1, 3, 3, 3), dtype=int)
    x[0, :, :, 0] = 1  # All class 0
    x[0, 1, 1, 0] = 0  # Switch middle to 1
    x[0, 1, 1, 1] = 1
    x[0, 2, 2, 0] = 0  # Switch LwRt to 1
    x[0, 2, 2, 2] = 1
    # label > adj
    # [0, 0, 0],  12, 4, 2
    # [0, 1, 0],   4, 0, 0
    # [0, 0, 2],   2, 0, 0
    gen = iter([(None, x), ])

    smooth = weighted_uniform_smoothing(
        n_classes,
        training_generator=gen,
        max_smoothing_magnitude=max_smoothing_magnitude,
        generator_length=1,
    )
    print(f"Smooth\n{smooth}")

    # TODO - define output
    expected = np.array([
        [0.94, 0.04,  0.02], 
        [0.06, 0.94,    0.],
        [0.06, 0.,    0.94],
    ])
    assert np.all(np.isclose(smooth, expected))


def test_weighted_adjacent_smooth():

    n_classes = 3
    max_smoothing_magnitude = 0.8
    x = np.zeros((1, 3, 3, 3), dtype=int)
    x[0, :, :, 0] = 1  # All class 0
    x[0, 1, 1, 0] = 0  # Switch middle to 1
    x[0, 1, 1, 1] = 1
    x[0, 2, 2, 0] = 0  # Switch LwRt to 1
    x[0, 2, 2, 2] = 1
    # label > adj
    # [0, 0, 0],  12, 4, 2
    # [0, 1, 0],   4, 0, 0
    # [0, 0, 2],   2, 0, 0
    gen = iter([(None, x), ])

    smooth = weighted_adjacent_smoothing(
        n_classes,
        training_generator=gen,
        max_smoothing_magnitude=max_smoothing_magnitude,
        generator_length=1,
    )
    print(f"Smooth\n{smooth}")

    # TODO - define output
    expected = np.array([
        [0.94, 0.04,  0.02], 
        [0.06, 0.94,    0.],
        [0.06, 0.,    0.94],
    ])
    assert np.all(np.isclose(smooth, expected))


if __name__ == "__main__":

    test_fixed_uniform_smooth()
    test_fixed_adjacent_smooth()
    # test_weighted_uniform_smooth()
    # test_weighted_adjacent_smooth()