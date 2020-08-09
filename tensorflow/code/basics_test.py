#!/usr/bin/env python3

from os import environ
environ['CUDA_VISIBLE_DEVICES'] = '6'
import tensorflow as tf
from tensorflow.compat.v1 import logging as tf_logger

import numpy as np


def disable_tensorflow_logging():
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf_logger.set_verbosity(tf_logger.ERROR)


# debug, info, warn, error, fatal
def enable_tensorflow_logging(level: int = tf_logger.INFO):
    environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    tf_logger.set_verbosity(level)


def test_gpu():
    print(tf.test.is_gpu_available())  # True


def test_cast():
    a = tf.cast(10, tf.float32)
    assert isinstance(a, tf.Tensor)
    a = tf.cast([10, 20], tf.float32)
    print(a)


def test_logging():

    def _test(msg):
        print(msg)
        tf_logger.debug('hello debug')
        tf_logger.info('hello info')
        tf_logger.warn('hello warn')
        tf_logger.error('hello error')
        tf_logger.fatal('hello fatal')
        print()

    enable_tensorflow_logging(tf_logger.DEBUG)
    _test('----debug----')

    enable_tensorflow_logging(tf_logger.INFO)
    _test('----info----')

    enable_tensorflow_logging(tf_logger.WARN)
    _test('----warn----')

    enable_tensorflow_logging(tf_logger.ERROR)
    _test('----error----')

    enable_tensorflow_logging(tf_logger.FATAL)
    _test('----fatal----')


def test_py_function():
    disable_tensorflow_logging()
    a = tf.constant([1., 2.])
    assert a.dtype == tf.float32

    b = tf.constant([10., 20.])

    def my_func(x, y):
        m = x + y
        return m, True, 10

    z = tf.py_function(my_func, [a, b], (tf.float32, tf.bool, tf.int32))
    assert isinstance(z, list)
    assert len(z) == 3
    assert z[0].dtype == tf.float32
    assert z[1].dtype == tf.bool
    assert z[2].dtype == tf.int32
    assert isinstance(z[0], tf.Tensor)
    assert isinstance(z[1], tf.Tensor)
    assert isinstance(z[2], tf.Tensor)

    with tf.Session() as sess:
        d = sess.run(z)
        np.testing.assert_array_equal(d[0], [11, 22])
        assert d[1] == True
        assert d[2] == 10


def test_stft():
    disable_tensorflow_logging()
    a = [1., -1., 3., 10, 13, 8, -9, 12, 20, -3, 11]
    assert len(a) == 11
    tf.reset_default_graph()

    b = tf.constant(value=a)
    # fft_length is set to frame_lenght by default.
    # frame_length is the num_fft_bins
    # frame_step is window_shift
    t = tf.contrib.signal.stft(
        signals=b,
        frame_length=4,
        frame_step=2,
        window_fn=lambda frame_length, dtype: (tf.contrib.signal.hann_window(frame_length, periodic=True, dtype=dtype)),
        pad_end=True)
    it = tf.contrib.signal.inverse_stft(
        stfts=t,
        frame_length=4,
        frame_step=2,
        window_fn=lambda frame_length, dtype: (tf.contrib.signal.hann_window(frame_length, periodic=True, dtype=dtype)))
    with tf.Session() as sess:
        tf_stft = sess.run(t)
        tf_istft = sess.run(it)
        import math
        assert tf_stft.shape[0] == math.ceil(len(a) / 2)  # 2 is frame_step
        assert tf_stft.shape[1] == 4 // 2 + 1  # num_fft_bins/2 + 1
        assert tf_stft.dtype == np.complex64
        print(tf_istft)

    tf.reset_default_graph()
    enable_tensorflow_logging()

    arr = np.array([
        [1, -1, 3, 10],
        [3, 10, 13, 8],
        [13, 8, -9, 12],
        [-9, 12, 20, -3],
        [20, -3, 11, 0],
        [11, 0, 0, 0],
    ])
    import scipy.signal
    np_stft = np.fft.fft(scipy.signal.windows.hann(4, sym=False) * arr, axis=-1)
    assert np_stft.shape[0] == 6
    assert np_stft.shape[1] == 4

    # note that numpy.fft.fft keeps all num_fft_bins
    # hann_window is DIFFERENT from the hanning_window

    print(tf_stft)
    print(np_stft)

    print(a)
    _, _, scipy_stft = scipy.signal.stft(a, nperseg=4, noverlap=2, boundary=None, padded=True, return_onesided=False)
    print(scipy_stft)


def test_floormod():
    disable_tensorflow_logging()
    tf.reset_default_graph()
    a = tf.math.floormod(11, 3)
    b = tf.math.floormod(3, 11)
    with tf.Session() as sess:
        c = sess.run(a)
        d = sess.run(b)
        assert c == 2  # it is the remainder of 11/3
        assert d == 3

    tf.reset_default_graph()
    enable_tensorflow_logging()


def test_pad():
    disable_tensorflow_logging()
    tf.reset_default_graph()

    a = np.arange(24).reshape(2, 3, 4)
    b = tf.pad(a, [[1, 1], [0, 0], [2, 2], [3, 3]])
    print(b.shape)
    with tf.Session() as sess:
        c = sess.run(b)
        print(c)

    tf.reset_default_graph()
    enable_tensorflow_logging()


def test_concat():
    disable_tensorflow_logging()
    tf.reset_default_graph()

    b = tf.concat([[1, 2], [3, 4]], axis=0)
    assert b.shape == (4,)

    with tf.Session() as sess:
        c = sess.run(b)
        np.testing.assert_array_equal(c, [1, 2, 3, 4])

    tf.reset_default_graph()
    enable_tensorflow_logging()


def pad_and_partition(tensor, segment_len):
    """ Pad and partition a tensor into segment of len segment_len
    along the first dimension. The tensor is padded with 0 in order
    to ensure that the first dimension is a multiple of segment_len.

    Tensor must be of known fixed rank

    :Example:

    >>> tensor = [[1, 2, 3], [4, 5, 6]]
    >>> segment_len = 2
    >>> pad_and_partition(tensor, segment_len)
    [[[1, 2], [4, 5]], [[3, 0], [6, 0]]]

    :param tensor:
    :param segment_len:
    :returns:
    """
    tensor_size = tf.math.floormod(tf.shape(tensor)[0], segment_len)
    pad_size = tf.math.floormod(segment_len - tensor_size, segment_len)
    padded = tf.pad(tensor, [[0, pad_size]] + [[0, 0]] * (len(tensor.shape) - 1))
    split = (tf.shape(padded)[0] + segment_len - 1) // segment_len
    return tf.reshape(padded, tf.concat([[split, segment_len], tf.shape(padded)[1:]], axis=0))


def test_pad_and_partition():
    disable_tensorflow_logging()
    tf.reset_default_graph()
    a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    a = tf.constant(a, dtype=tf.float32)
    b = pad_and_partition(a, 2)
    print(b.shape)
    with tf.Session() as sess:
        c = sess.run(b)
        print(c)

    tf.reset_default_graph()
    enable_tensorflow_logging()


def test_reduce_sum():
    disable_tensorflow_logging()
    tf.reset_default_graph()
    a = [
        [1, 2, 3],
        [4, 5, 6.],
    ]
    b = [
        [10, 20, 30],
        [40, 50, 60],
    ]

    c = tf.reduce_sum([a, b], axis=0)
    with tf.Session() as sess:
        f = sess.run(c)
        np.testing.assert_array_equal(f, [[11., 22, 33], [44, 55, 66]])

    tf.reset_default_graph()
    enable_tensorflow_logging()


def main():
    #  test_cast()
    #  test_gpu()
    #  test_logging()
    #  test_py_function()
    #  test_stft()
    #  test_floormod()
    #  test_pad()
    #  test_concat()
    #  test_pad_and_partition()
    test_reduce_sum()
    pass


if __name__ == '__main__':
    main()
