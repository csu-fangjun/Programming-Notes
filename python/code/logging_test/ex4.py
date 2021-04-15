#!/usr/bin/env python3

import logging

logging.info('no output')
logging.basicConfig(level=logging.INFO)
logging.info('also output')

# CAUTION:
# logging.basicConfig should be called before any invocation of logging.xxx()
# Any subsequence calls of it are ignored!
