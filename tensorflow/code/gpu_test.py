#!/usr/bin/env python3

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
