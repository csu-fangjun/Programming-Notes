#!/usr/bin/python3

# refer to https://github.com/tensorflow/tensorflow/issues/27045
# to disable tensorflow warnings
import logging
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
# logging.disable(logging.WARNING) # either is ok

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import tensorflow as tf


def test_dataset():
    '''
    Make a dataset from a generator and iterate through the dataset
    '''
    a = range(3)
    dataset = tf.data.Dataset.from_generator(lambda: (i for i in a), tf.int32)
    iter = dataset.make_initializable_iterator()

    el = iter.get_next()
    with tf.Session() as sess:
        sess.run(iter.initializer)
        try:
            while True:
                b = sess.run(el)
                assert isinstance(b, np.int32)
                assert b in [0, 1, 2]
        except tf.errors.OutOfRangeError:
            pass

    def parse(x):
        return {'name': x}

    d = dataset.map(parse)

    iter = d.make_initializable_iterator()
    el = iter.get_next()
    with tf.Session() as sess:
        sess.run(iter.initializer)
        try:
            while True:
                b = sess.run(el)
                assert isinstance(b, dict)
                assert 'name' in b
                assert isinstance(b['name'], np.int32)
                assert b['name'] in [0, 1, 2]
        except tf.errors.OutOfRangeError:
            pass


def test_config_proto():
    '''
    Refer to https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto

    Set `allow_grow` to True so that it will not consume all GPU memory.
    '''
    p = tf.ConfigProto()
    p.gpu_options.allow_grow = True
    tf.Session(config=p)


def test_global_step():
    with tf.Graph().as_default() as g:
        tensor = tf.train.get_or_create_global_step()
        inc_one = tensor.assign(tensor + 1)
        with tf.Session() as sess:
            # global step is a variable, so we have to initialize it
            sess.run(tf.global_variables_initializer())
            t = sess.run(tensor)
            assert t == 0

            sess.run(inc_one)
            t = sess.run(tensor)
            assert t == 1

            t = sess.run(tensor)
            assert t == 1

            # increase it the second time
            sess.run(inc_one)
            t = sess.run(tensor)
            assert t == 2


def test_variable_scope():
    with tf.variable_scope('test'):
        t = tf.constant([1, 2, 3], name='hello')
        assert t.name == 'test/hello:0'


def test_one_hot():
    t = tf.constant([0, 1, 2, 3, 2], dtype=tf.int32)
    num_classes = 4
    one = tf.one_hot(t, num_classes)
    with tf.Session() as sess:
        labels = sess.run(one)
        assert isinstance(labels, np.ndarray)
        ground_truth = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ])
        np.testing.assert_array_equal(labels, ground_truth)


def test_sequence_mask():
    seq_len = tf.constant([2, 1, 3], dtype=tf.int32)
    mask = tf.sequence_mask(seq_len)
    with tf.Session() as sess:
        val = sess.run(mask)
        assert isinstance(val, np.ndarray)
        ground_truth = np.array([
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ])
        np.testing.assert_array_equal(val, ground_truth)

    seq_len = tf.constant([2, 1, 3], dtype=tf.int32)
    mask = tf.sequence_mask(seq_len, maxlen=4)
    with tf.Session() as sess:
        val = sess.run(mask)
        assert isinstance(val, np.ndarray)
        ground_truth = np.array([
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, False],
        ])
        np.testing.assert_array_equal(val, ground_truth)


def test_equal():
    #                0  1  2  3  4  5
    a = tf.constant([1, 1, 2, 3, 2, 2], dtype=tf.int32)
    b = tf.equal(a, 2)
    data = tf.constant([
        [1, 2],  # 0
        [3, 4],  # 1
        [5, 6],  # 2
        [7, 8],  # 3
        [9, 10],  # 4
        [11, 12],  # 5
    ])
    index = tf.where(b)
    neg = tf.gather_nd(data, index)
    with tf.Session() as sess:
        val_b = sess.run(b)
        #                                       0     1      2     3     4     5
        np.testing.assert_array_equal(val_b, [False, False, True, False, True, True])
        val_index = sess.run(index)
        np.testing.assert_array_equal(val_index, [[2], [4], [5]])  # note it is 2-D

        val_neg = sess.run(neg)
        np.testing.assert_array_equal(val_neg, [[5, 6], [9, 10], [11, 12]])


def test_reduce():
    a = tf.constant([1, 2], dtype=tf.float32)
    b = tf.reduce_mean(a)
    with tf.Session() as sess:
        c = sess.run(b)
        assert c == 1.5

    b = tf.reduce_min(a)
    with tf.Session() as sess:
        c = sess.run(b)
        assert c == 1

    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    b = tf.reduce_mean(a)
    with tf.Session() as sess:
        c = sess.run(b)
        assert c == 2.5  # note that c is a scalar!


def test_tf_map():
    b = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8]])

    def fn(i):
        print(i)
        return b[i:i + 2, :]

    pad = tf.map_fn(fn, tf.range(3))

    data = tf.constant([
        [[10, 20]],
        [[30, 40]],
        [[50, 60]],
    ])

    d = tf.concat([pad, data], axis=1)
    with tf.Session() as sess:
        pad = sess.run(pad)
        assert pad.shape == (3, 2, 2)
        np.testing.assert_array_equal(pad, [
            [[1, 2], [3, 4]],
            [[3, 4], [5, 6]],
            [[5, 6], [7, 8]],
        ])

        val_d = sess.run(d)
        print(val_d)
        assert val_d.shape == (3, 3, 2)  # from (3, 1, 2) to (3, 3, 2)
        np.testing.assert_array_equal(val_d, [
            [[1, 2], [3, 4], [10, 20]],
            [[3, 4], [5, 6], [30, 40]],
            [[5, 6], [7, 8], [50, 60]],
        ])


def test_tf_tile():
    a = tf.constant([1, 2, 3], dtype=tf.float32)
    b = tf.sequence_mask(a)
    b = tf.expand_dims(b, -1)
    c = tf.tile(b, [1, 1, 3])
    with tf.Session() as sess:
        val_b = sess.run(b)
        assert val_b.shape == (3, 3, 1)
        np.testing.assert_array_equal(val_b,
                                      [[[True], [False], [False]], [[True], [True], [False]], [[True], [True], [True]]])

        val_c = sess.run(c)
        assert val_c.shape == (3, 3, 3)
        np.testing.assert_array_equal(val_c, [[[True, True, True], [False, False, False], [False, False, False]],
                                              [[True, True, True], [True, True, True], [False, False, False]],
                                              [[True, True, True], [True, True, True], [True, True, True]]])

    d = tf.sequence_mask(a)
    d = tf.expand_dims(d, 1)
    e = tf.tile(d, [1, 3, 1])
    with tf.Session() as sess:
        val_d = sess.run(d)
        assert val_d.shape == (3, 1, 3)

        val_e = sess.run(e)
        assert val_e.shape == (3, 3, 3)
        print(val_e)
        np.testing.assert_array_equal(val_e, [
            [[True, False, False], [True, False, False], [True, False, False]],
            [[True, True, False], [True, True, False], [True, True, False]],
            [[True, True, True], [True, True, True], [True, True, True]],
        ])


def test_gather():
    a = tf.constant([
        [
            [1, 2],
            [3, 4],
        ],
        [
            [5, 6],
            [7, 8],
        ],
        [
            [9, 10],
            [11, 12],
        ],
    ])

    b = tf.gather(a, [1, 0, 2])
    with tf.Session() as sess:
        val_b = sess.run(b)
        np.testing.assert_array_equal(val_b, [
            [
                [5, 6],
                [7, 8],
            ],
            [
                [1, 2],
                [3, 4],
            ],
            [
                [9, 10],
                [11, 12],
            ],
        ])
        print(val_b)


def test_stack():
    a = tf.constant([1, 2, 3], dtype=tf.float32)
    b = tf.constant([10, 20, 30], dtype=tf.float32)
    c = tf.stack([a, b], 0)
    with tf.Session() as sess:
        val_c = sess.run(c)
        np.testing.assert_array_equal(val_c, [[1, 2, 3], [10, 20, 30]])

    d = tf.stack([a, b], 1)
    with tf.Session() as sess:
        val_d = sess.run(d)
        np.testing.assert_array_equal(val_c, [[1, 10], [2, 20], [3, 30]])


def main():
    #  test_dataset()
    #  test_config_proto()
    #  test_global_step()
    #  test_variable_scope()
    #  test_one_hot()
    #  test_sequence_mask()
    #  test_equal()
    test_reduce()
    #  test_tf_map()
    #  test_tf_tile()
    #  test_gather()
    #  test_stack()

    pass


if __name__ == '__main__':
    main()
