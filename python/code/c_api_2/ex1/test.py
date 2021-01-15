#!/usr/bin/env python3

import sys
sys.path.insert(0, 'build/lib.linux-x86_64-3.8')

import spam

if __name__ == '__main__':
    spam.system('ls -l')
    error = spam.error
    # error is a type `spam.Error`
    #
    # we have used PyErr_NewException("spam.error", nullptr, nullptr);
    assert error.__module__ == 'spam'
    e = error()
    print(e.__class__, e.__module__)
    assert isinstance(e, spam.error)
