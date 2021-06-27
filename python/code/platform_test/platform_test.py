#!/usr/bin/env python3

import platform

# see https://github.com/PyTorchLightning/pytorch-lightning/blob/master/tests/collect_env_details.py#L51
# see https://docs.python.org/3/library/platform.html


def info_system():
    ans = {
        'OS': platform.system(),  # Darwin, Linux
        'Architecture': platform.architecture(),  # Darwin, Linux
    }
    print(ans)


def test1():
    print(platform.architecture())  # ('64bit', 'ELF')
    print(platform.machine())  # x86_64
    print(platform.node())  # name of the machine, e.g., ubuntu
    print(platform.platform()
         )  # Linux-4.4.0-170-generic-x86_64-with-Ubuntu-16.04-xenial
    print(platform.platform(
        terse=True))  # Linux-4.4.0-170-generic-x86_64-with-glibc2.9
    print(platform.processor())  # x86_64
    print(platform.python_build())  # ('default', 'Oct 8 2019 13:06:37')
    print(platform.python_compiler())  # GCC 5.4.0 20160609
    print(platform.python_branch())  #
    print(platform.python_implementation())  # CPython
    print(platform.python_revision())  #
    print(platform.python_version_tuple())  # ('3', '5', '2')
    print(platform.release())  # 4.4.0-170-generic
    print(platform.system())  # Linux
    print(platform.version())  # #199-Uubuntu SMP Thu Nov 14 01:45:04 UTC 2019
    # yapf: disable
    print(platform.uname())  # uname_result(system='Linux', node='ubuntu', release='4.4.0-170-generic', version='#199-Ubuntu SMP Thu Nov 14 01:45:04 UTC 2019', machine='x86_64', processor='x86_64')
    # yapf: enable


def main():
    info_system()


if __name__ == '__main__':
    main()
