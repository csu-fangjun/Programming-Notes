#!/usr/bin/env python3

import click


@click.command()
def hello():
    print('hello world')


if __name__ == '__main__':
    hello()
'''
See https://click.palletsprojects.com/en/7.x/quickstart/

Usage:
    ./1.py --help
    ./1.py

Note(fangjun): the only usage of click here is to add an help option for 1.py
'''
