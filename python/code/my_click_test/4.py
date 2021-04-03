#!/usr/bin/env python3

import click


@click.command()
@click.option('--count', default=1, help='number of greetings')
@click.argument('name')
def hello(count, name):
    print('count is', count)
    print('name is', name)


if __name__ == '__main__':
    hello()
'''
Usage:
    ./4.py --help
    ./4.py tom
    ./4.py --count=2 tom
    ./4.py --count 2 tom
    ./4.py tom --count 2
    ./4.py tom --count=2

'''
