#!/usr/bin/env python3

from setuptools import setup

setup(
    name='h1',
    version='0.1',
    py_modules=['h1'],
    install_requires=[
        'Click',
    ],
    entry_points='''
            [console_scripts]
            h1=h1:cli
        ''',
)
