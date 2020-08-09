#!/usr/bin/env python3

import argparse


def test1():
    parser = argparse.ArgumentParser(description='this is a description')

    # integers is a positional argument
    # usage: ./a.out 1 2 3
    #
    # metavar is **only** used in the help message
    # there is no `dest` here, so the value is saved in the field `integers`
    parser.add_argument('integers', metavar='N', type=int, nargs='+', help='a list of integers')

    args = parser.parse_args('1'.split())
    assert isinstance(args.integers, list)
    assert args.integers == [1]

    args = parser.parse_args('1 2 3'.split())
    assert isinstance(args.integers, list)
    assert args.integers == [1, 2, 3]


def test2():
    parser = argparse.ArgumentParser('test2', description='test subparser')
    parser.add_argument('--top-level', help='option in the parent')

    # we can use `args.command` to differentiate which subcommand is used
    subparsers = parser.add_subparsers(dest='command')

    parser1 = subparsers.add_parser('train', description='parser for training', help='train help')
    parser1.add_argument('--train-dir', help='train dir')
    parser1.add_argument('dir', help='output dir')

    parser1 = subparsers.add_parser('eval', description='parser for eval', help='eval help')
    parser1.add_argument('--eval-dir', help='eval dir')
    parser1.add_argument('dir', help='output dir')

    args = parser.parse_args('--top-level 2 train --train-dir abc out'.split())
    assert args.command == 'train'
    assert args.top_level == '2'
    assert args.train_dir == 'abc'
    assert args.dir == 'out'

    args = parser.parse_args()


def main():
    test1()
    test2()


if __name__ == '__main__':
    main()
