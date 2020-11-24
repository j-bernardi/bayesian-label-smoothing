import numpy as np

from utils import adjacency_from_generator, class_sums_from_generator

def make_test_cases():

    labels = []
    adjacencies = []

    ## 1 - 0s everywhere
    label = np.zeros((1, 3, 3, 2))
    label[0, :, :, 0] = 1
    expected = {
        "count": np.array([9, 0]),
        "adj": np.array([
            [24, 0],
            [0,  0]
        ])
    }
    labels.append(label)
    adjacencies.append(expected)

    ## 2 - 1 at top edge
    label = np.zeros((1, 3, 3, 2))
    label[0, :, :, 1] = 0
    label[0, 0, 1, 0] = 0
    label[0, 0, 1, 1] = 1
    expected = {
        "count": np.array([8, 1]),
        "adj": np.array([
            [18, 3],
            [3,  0]
        ])
    }
    labels.append(label)
    adjacencies.append(expected)

    ## 3 - 1 at right edge
    label = np.zeros((1, 3, 3, 2))
    label[0, :, :, 1] = 0
    label[0, 1, 2, 0] = 0
    label[0, 1, 2, 1] = 1
    expected = {
        "count": np.array([8, 1]),
        "adj": np.array([
            [18, 3],
            [3,  0]
        ])
    }
    labels.append(label)
    adjacencies.append(expected)

    ## 4 - central class 1
    # label > (one hot)
    # 0 0 0
    # 0 1 0
    # 0 0 0
    label = np.zeros((1, 3, 3, 2))
    label[0, :, :, 1] = 0
    label[0, 1, 1, 0] = 0   # swap class in centre
    label[0, 1, 1, 1] = 1

    # Adjacency matrix
    # (0, 0), (0, 1)
    # (1, 0), (1, 1) >
    expected = {
        "count": np.array([8, 1]),
        "adj": np.array([
            [16, 4],
            [4,  0]
        ])
    }
    labels.append(label)
    adjacencies.append(expected)

    return labels, adjacencies


def test_sums_end_to_end():
    print("\nTesting sums")
    for label, expected in zip(*make_test_cases()):

        sums = class_sums_from_generator(2, iter([(0, label),]), 1)
        print("Sums", sums)
        print("Expected", expected["count"])

        assert np.all(sums == expected["count"])


def test_adjacency_end_to_end():
    print("\nTesting adj")
    for label, expected in zip(*make_test_cases()):
        print(f"Label\n{np.argmax(label, -1)}")
        print(f"Expected adj\n{expected['adj']}")

        adj = adjacency_from_generator(2, iter([(0, label),]), 1)
        print(f"Adjacency matrix:\n{adj}")

        assert np.all(adj == expected["adj"])

if __name__ == "__main__":
    test_sums_end_to_end()
    test_adjacency_end_to_end()
