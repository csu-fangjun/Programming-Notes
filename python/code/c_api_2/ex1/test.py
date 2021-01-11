#!/usr/bin/env python3

import sys
sys.path.insert(0, 'build/lib.linux-x86_64-3.8')

import spam

if __name__ == '__main__':
    spam.system('ls -l')
