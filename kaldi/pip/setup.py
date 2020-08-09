# Copyright 2020 Mobvoi Inc. All Rights Reserved.
# Author: fangjun.kuang@mobvoi.com (Fangjun Kuang)

from setuptools import setup

setup(name='kaldi',
      version='v20200316',
      description='python wrapper for kaldi ',
      author='Fangjun Kuang ',
      author_email='fangjun.kuang@mobvoi.com',
      packages=['kaldi'],
      include_package_data=True,
      install_requires=['numpy'],
      keywords='kaldi',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
      ],
      long_description='''this is a python wrapper
for kaldi using pybind11.

You do not need to install kaldi to use it. Everything
you need is included in this pip package. It currently
support CPU only.
''')
