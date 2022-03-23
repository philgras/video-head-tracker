from typing import *
from torchvision.transforms.functional import gaussian_blur


class DecayScheduler:
    def __init__(
        self, start_val, end_val, start_step: int, end_step: int, geometric=False
    ):
        assert end_step > start_step
        step_diff = end_step - start_step
        if geometric:
            self._slope = (end_val / (start_val + 1e-14)) ** (1 / step_diff)
        else:
            self._slope = (end_val - start_val) / step_diff
        self._start_step = start_step
        self._end_step = end_step
        self._start_val = start_val
        self._end_val = end_val
        self._geometric = geometric

    def get(self, epoch):
        if epoch < self._start_step:
            return self._start_val
        elif epoch >= self._end_step:
            return self._end_val
        else:
            step = epoch - self._start_step
            if self._geometric:
                return self._start_val * self._slope ** step
            else:
                return self._start_val + self._slope * step


def closest_odd_int(x):
    return int((x // 2) * 2 + 1)


def blur_tensors(*tensors: List, sigma=0.0):
    tensors = list(tensors)
    if sigma == 0:
        return tensors
    else:
        kernel_size = closest_odd_int(sigma * 6)
        for i in range(len(tensors)):
            tensors[i] = gaussian_blur(tensors[i], kernel_size=kernel_size, sigma=sigma)
        return tensors
