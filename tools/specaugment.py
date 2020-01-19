import random
import numpy as np


def freq_mask(spec, l_mask, n_mask):
    """Masking frequency

    @param spec:                spectogram
    @param l_mask:              length of each mask
    @param n_mask:              number of masks
    @return:                    frequency masked spectogram
    """
    cloned = spec

    num_mel_channels = cloned.shape[1]
    fs = np.random.randint(0, l_mask, size=(n_mask, 2))

    for f, mask_end in fs:
        f_zero = random.randrange(0, num_mel_channels - f)
        mask_end += f_zero

        if f_zero == f_zero + f:
            continue

        cloned[:, f_zero:mask_end] = 0

    return cloned


def time_mask(spec, l_mask, n_mask):
    """Masking time

        @param spec:                spectogram
        @param l_mask:              length of each mask
        @param n_mask:              number of masks
        @return:                    time masked spectogram
        """
    cloned = spec

    len_spectro = cloned.shape[0]
    ts = np.random.randint(0, l_mask, size=(n_mask, 2))

    for t, mask_end in ts:
        if len_spectro - t <= 0:
            continue

        t_zero = random.randrange(0, len_spectro - t)

        if t_zero == t_zero + t:
            continue

        mask_end += t_zero
        cloned[t_zero:mask_end] = 0

    return cloned


def spec_augment(x, max_freq_width=0.15, n_freq_mask=2, max_time_width=0.15, n_time_mask=2):
    """Perform partial specaugment

    @param x:                               input spectogram
    @param max_freq_width:                  frac of masking width per mask
    @param n_freq_mask:                     number of width masks
    @param max_time_width:                  frac of masking time per mask
    @param n_time_mask:                     number of time masks
    @return:
    """

    max_time_width = int(np.round(max_time_width * x.shape[0]))
    max_freq_width = int(np.round(max_freq_width * x.shape[1]))

    n_freq_mask = random.randint(0, n_freq_mask)
    n_time_mask = random.randint(0, n_time_mask)

    x = freq_mask(x, max_freq_width, n_freq_mask)
    x = time_mask(x, max_time_width, n_time_mask)

    return x
