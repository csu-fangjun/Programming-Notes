#!/usr/bin/env python3

import click


@click.command()
def cli():
    print('hello cli')


'''
See https://click.palletsprojects.com/en/7.x/setuptools/

Usage:
    pip install --editable .

It will create a file /path/to/venv/bin/h1
'''
