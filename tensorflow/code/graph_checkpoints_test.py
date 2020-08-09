#!/usr/bin/env python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.compat.v1 import logging as tf_logger
tf_logger.set_verbosity(tf_logger.ERROR)

import numpy as np


def build_model():
    x = tf.placeholder(tf.float32, shape=(None,), name='x')
    y = tf.placeholder(tf.float32, shape=(None,), name='y')
    with tf.variable_scope('regression'):
        w = tf.Variable(np.random.normal(), name='W')
        b = tf.Variable(np.random.normal(), name='b')
        y_pred = tf.add(tf.multiply(w, x), b)

        loss = tf.reduce_mean(tf.square(y_pred - y))
    return loss


# First get all the operations;
# then get the output of each operation
def print_all_tensor_names(g: tf.Graph) -> None:
    for op in g.get_operations():
        assert isinstance(op, tf.Operation)
        assert op.graph == g
        print('name: {}'.format(op.name))
        assert op.values() == tuple(op.outputs)
        for tensor in op.inputs:
            assert isinstance(tensor, tf.Tensor)
            print(' in', tensor.name)
        for tensor in op.outputs:
            assert isinstance(tensor, tf.Tensor)
            print(' out', tensor.name)


# print the value of a given tensor
def print_tensors(g: tf.Graph) -> None:
    with tf.Session(graph=g) as sess:
        sess.run(tf.initializers.global_variables())
        w0 = g.get_tensor_by_name('regression/W:0').eval()

        w1 = g.get_tensor_by_name('regression/W/read:0').eval()
        assert w0 == w1


def load_graph(filename: str) -> tf.Graph:
    g = tf.Graph()
    with g.as_default():
        with open(filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')
    return g


def save_for_tensorboard(logdir: str,
                         graph: tf.Graph = None,
                         graph_def: tf.GraphDef = None) -> None:
    writer = tf.summary.FileWriter(logdir=logdir,
                                   graph=graph,
                                   graph_def=graph_def)
    writer.flush()
    writer.close()


# test save and load graph
def save_graph(g: tf.Graph) -> None:
    tf.train.write_graph(g, logdir='./', name='g.pb', as_text=False)

    # now load it
    g = load_graph('g.pb')
    # now save it to text format.
    tf.train.write_graph(g, logdir='./', name='g2.pb', as_text=True)


def freeze_graph(g: tf.Graph) -> None:
    with tf.Session(graph=g) as sess:
        sess.run(tf.initializers.global_variables())
        # convert_variables_to_constants
        # it keeps only parts of the graph that is accessible and coaccessible
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=g.as_graph_def(),
            output_node_names=['regression/Mean'])
        tf.train.write_graph(frozen_graph_def,
                             logdir='./',
                             name='f.pb',
                             as_text=False)

        save_for_tensorboard(logdir='./frozen', graph_def=frozen_graph_def)

    g = load_graph('f.pb')
    print_all_tensor_names(g)


def save_checkpoints(g: tf.Graph):
    with tf.Session(graph=g) as sess:
        sess.run(tf.initializers.global_variables())
        w = g.get_tensor_by_name('regression/W/read:0').eval()
        print('in save:', w)

        saver = tf.train.Saver()
        saver.save(sess=sess, save_path='./abc/my_model', global_step=10)
        # it saves the checkpoint **files** to the directory abc.
        # `my_model` is the prefix
        # it will create the following files
        #  checkpoint                       (79-byte)
        # my_model-10.data-00000-of-00001   (8-byte)
        # my_model-10.index                 (145-byte)
        # my_model-10.meta                  (5.0-KB)
        #
        #  The checkpoint file contains the following two lines
        #     model_checkpoint_path: "my_model-10"
        #     all_model_checkpoint_paths: "my_model-10"
        #


def restore_checkpoints1():
    # load the graph from the meta file saved in the checkpoints
    g = tf.Graph()
    with g.as_default():
        # import_meta_graph will fill the current default graph

        tf.train.import_meta_graph('./abc/my_model-10.meta')
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './abc/my_model-10')
        w = g.get_tensor_by_name('regression/W/read:0').eval()
        print('in load:', w)


def restore_checkpoints2():
    g = tf.Graph()
    with g.as_default():
        w = tf.get_variable('regression/W', initializer=1., dtype=tf.float32)
    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, './abc/my_model-10')
        w = g.get_tensor_by_name('regression/W/read:0').eval()
        print('in load2:', w)

    with tf.Session(graph=g) as sess:
        sess.run(tf.initializers.global_variables())
        w = g.get_tensor_by_name('regression/W/read:0').eval()
        print('in load2, before load,', w)
        saver = tf.train.Saver()
        saver.restore(sess, './abc/my_model-10')
        w = g.get_tensor_by_name('regression/W/read:0').eval()
        print('in load2:', w)


def test_conv2d():
    from tensorflow.keras.layers import Conv2D
    shape = [1, 20, 30, 1]  # NHWC
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(20200711)
        placeholder = tf.placeholder(tf.float32, shape=shape)
        x = Conv2D(filters=3,
                   kernel_size=(5, 5),
                   strides=(2, 2),
                   padding="same",
                   bias_initializer="glorot_uniform")(placeholder)
        for op in g.get_operations():
            for tensor in op.outputs:
                print(tensor.name, tensor.shape)
    with tf.Session(graph=g) as sess:
        sess.run(tf.initializers.global_variables())

        w = g.get_tensor_by_name('conv2d/kernel/Read/ReadVariableOp:0').eval()
        b = g.get_tensor_by_name('conv2d/bias/Read/ReadVariableOp:0').eval()
        print(w.shape, b.shape)
        print(w.sum(), b, b.sum())
        #  print(w.shape, b.shape, w.sum(), b.sum())
        #  print(w, b)

        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=g.as_graph_def(),
            output_node_names=['conv2d/BiasAdd'])
    g2 = tf.Graph()
    with g2.as_default():
        tf.graph_util.import_graph_def(frozen_graph_def, name='')
    for op in g2.get_operations():
        for tensor in op.outputs:
            print(tensor.name, tensor.shape)
    with tf.Session(graph=g2) as sess:
        w = g2.get_tensor_by_name('conv2d/kernel:0').eval()
        b = g2.get_tensor_by_name('conv2d/bias:0').eval()
        print(w.shape, b.shape)
        print(w.sum(), b, b.sum())
        print('here', sess.run('conv2d/kernel:0').shape)
    op = g2.get_operation_by_name('conv2d/Conv2D')
    print(op, type(op))
    print(op.get_attr('dilations'))
    print(op.get_attr('strides'))
    print(op.get_attr('padding').decode())


def test():
    g = tf.Graph()
    with g.as_default():
        loss = build_model()
    assert loss.graph == g
    #  print_all_tensor_names(g)
    #  print_tensors(g)
    #  save_graph(g)
    #  freeze_graph(g)
    #  save_for_tensorboard(logdir='./', graph=g)
    save_checkpoints(g)
    restore_checkpoints1()
    restore_checkpoints2()


def main():
    #  test()
    test_conv2d()


if __name__ == '__main__':
    np.random.seed(123)
    main()
