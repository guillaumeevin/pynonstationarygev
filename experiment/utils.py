import numpy as np


def average_smoothing_with_sliding_window(x, y, window_size_for_smoothing):
    # Average on windows of size 2*M+1 (M elements on each side)
    kernel = np.ones(window_size_for_smoothing) / window_size_for_smoothing
    y = np.convolve(y, kernel, mode='valid')
    assert window_size_for_smoothing % 2 == 1
    if window_size_for_smoothing > 1:
        nb_to_delete = int(window_size_for_smoothing // 2)
        x = np.array(x)[nb_to_delete:-nb_to_delete]
    assert len(x) == len(y), "{} vs {}".format(len(x), len(y))
    return x, y
