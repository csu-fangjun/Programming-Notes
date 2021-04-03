#!/usr/bin/env python3

import click


@click.group()
def cli():
    pass


@click.command()
def begin():
    print('in begin')


@click.command()
def end():
    print('in end')


if __name__ == '__main__':
    cli.add_command(begin)
    cli.add_command(end)
    cli()
'''
See https://click.palletsprojects.com/en/7.x/quickstart/

Usage:
    ./2.py
    ./2.py --help
    ./2.py begin
    ./2.py end
'''
