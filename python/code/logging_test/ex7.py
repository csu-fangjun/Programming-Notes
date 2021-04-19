#!/usr/bin/env python3

import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s:%(message)s')
logging.debug('debug')
logging.info('info')
logging.warning('warning')