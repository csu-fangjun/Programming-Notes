#!/usr/bin/env python3

import click


@click.group()
def cli():
    pass


@cli.command()
def begin():
    print('in begin')


@cli.command()
def end():
    print('in end')


if __name__ == '__main__':
    cli()
'''
See https://click.palletsprojects.com/en/7.x/quickstart/

Usage:
    ./3.py
    ./3.py --help
    ./3.py begin
    ./3.py end
'''
