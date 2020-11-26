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
    expected = np.array([
        [0.95, 0.025, 0.025],
        [0.025, 0.95, 0.025],
        [0.025, 0.025, 0.95],
    ])
    print(f"Expected\n{expected}")

    smooth = fixed_uniform_smoothing(
        n_classes,
        fixed_smoothing_magnitude=magnitude,
    )

    print(f"Smooth\n{smooth}")

    assert np.all(np.isclose(smooth, expected)),\
        np.isclose(smooth, expected)


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
    
    expected = np.array([
        [0.94, 0.04,  0.02], 
        [0.06, 0.94,    0.],
        [0.06, 0.,    0.94],
    ])
    print(f"Expected\n{expected}")

    smooth = fixed_adjacent_smoothing(
        n_classes,
        fixed_smoothing_magnitude=fixed_smoothing_magnitude,
        training_generator=gen,
        generator_length=1,
    )
    print(f"Smooth\n{smooth}")

    assert np.all(np.isclose(smooth, expected)),\
        np.isclose(smooth, expected)


def test_weighted_uniform_smooth():

    n_classes = 3
    max_smoothing_magnitude = 0.90
    x = np.zeros((1, 3, 3, 3), dtype=int)
    x[0, :, :, 0] = 1  # All class 0
    x[0, 0, 1, 0] = 0  # Switch top-middle to 1
    x[0, 0, 1, 1] = 1
    x[0, 0, 2, 0] = 0  # Switch top-right to 1
    x[0, 0, 2, 1] = 1
    x[0, 2, 2, 0] = 0  # Switch low-right to 1
    x[0, 2, 2, 2] = 1
    # label > adj
    # [0, 1, 1],  12, 3, 2
    # [0, 0, 0],   3, 2, 0
    # [0, 0, 2],   2, 0, 0
    gen = iter([(None, x), ])
    # max_mag + (self-adj / (sum of others)) * (1-max_mag)

    adj0 = (1. - max_smoothing_magnitude) * 12./17.
    r0 = 1. - (max_smoothing_magnitude + adj0)
    expected = np.array([
        [0.90+adj0, r0*0.5, r0*0.5], 
        [0.03,        0.94,   0.03],
        [0.05,        0.05,   0.90],
    ])
    print(f"Expected\n{expected}")

    smooth = weighted_uniform_smoothing(
        n_classes,
        training_generator=gen,
        max_smoothing_magnitude=max_smoothing_magnitude,
        generator_length=1,
    )
    print(f"Smooth\n{smooth}")

    assert np.all(np.isclose(smooth, expected)),\
        np.isclose(smooth, expected)


def test_weighted_adjacent_smooth():

    n_classes = 3
    max_smoothing_magnitude = 0.90
    x = np.zeros((1, 3, 3, 3), dtype=int)
    x[0, :, :, 0] = 1  # All class 0
    x[0, 0, 1, 0] = 0  # Switch top-middle to 1
    x[0, 0, 1, 1] = 1
    x[0, 0, 2, 0] = 0  # Switch top-left to 1
    x[0, 0, 2, 1] = 1
    x[0, 2, 2, 0] = 0  # Switch bottom-right to 1
    x[0, 2, 2, 2] = 1
    # label > adj
    # [0, 1, 1],  12, 3, 2
    # [0, 0, 0],   3, 2, 0
    # [0, 0, 2],   2, 0, 0
    gen = iter([(None, x), ])
    # max_mag + (self-adj / (sum of others)) * (1-max_mag)
    adj0 =  (1. - max_smoothing_magnitude) * 12./17.
    r0 = 1. - (max_smoothing_magnitude + adj0)
    expected = np.array([
        [0.90+adj0, r0*3./5., r0*2./5.], 
        [0.06,          0.94,       0.],
        [0.10,            0.,     0.90],
    ])
    print(f"Expected\n{expected}")

    smooth = weighted_adjacent_smoothing(
        n_classes,
        training_generator=gen,
        max_smoothing_magnitude=max_smoothing_magnitude,
        generator_length=1,
    )
    print(f"Smooth\n{smooth}")

    assert np.all(np.isclose(smooth, expected)), np.isclose(smooth, expected)


if __name__ == "__main__":
    test_fixed_uniform_smooth()
    test_fixed_adjacent_smooth()
    test_weighted_uniform_smooth()
    test_weighted_adjacent_smooth()
