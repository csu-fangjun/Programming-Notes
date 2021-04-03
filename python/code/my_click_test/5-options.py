#!/usr/bin/env python3

import click


@click.command()
@click.option('-s', '--string-to-print', default=1, show_default=True)
def echo(string_to_print):
    print('s is', string_to_print)


if __name__ == '__main__':
    echo()
'''
Refer to https://click.palletsprojects.com/en/7.x/options/
'''
