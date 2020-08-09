#!/usr/bin/env python3
r'''
this file demonstrates the useage of pysoundfile.

To install it, use:

    pip install soundfile

'''

import os

import numpy as np
import soundfile as sf


def main():
    filename = '/tmp/test.wav'
    if not os.path.isfile(filename):
        print('{} does not exist!'.format(filename))
        return
    data, samplerate = sf.read(filename)

    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float64
    assert -1 <= np.min(data) <= 1
    assert -1 <= np.max(data) <= 1

    assert isinstance(samplerate, int)

    data, samplerate = sf.read(filename, dtype='float32')

    assert isinstance(data, np.ndarray)
    assert data.dtype == np.float32
    assert -1 <= np.min(data) <= 1
    assert -1 <= np.max(data) <= 1

    assert isinstance(samplerate, int)


if __name__ == '__main__':
    main()
