#!/usr/bin/env python3

import logging
logging.basicConfig(filename='ex2.log', level=logging.DEBUG)
logging.debug('This message should go to the log file')
logging.info('So should this')
logging.warning('And this too')
logging.info('Also this')
