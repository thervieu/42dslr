import numpy as np
import math

def data_spliter(x, y, proportion):
    """
    Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible shapes.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    def shuffle_in_unison(a, b):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    if isinstance(x, np.ndarray) is False or isinstance(y, np.ndarray) is False or isinstance(proportion, float) is False:
        return None
    if x.size == 0 or y.size == 0 or proportion <= 0 or proportion >= 1:
        return None
    if x.shape[0] != y.shape[0]:
        return None
    shuffle_in_unison(x, y)
    return (x[:math.floor(proportion * len(x))], x[math.floor(proportion * len(x)) - len(x):], y[:math.floor(proportion * len(x))], y[math.floor(proportion * len(x)) - len(x):])
