#!/usr/bin/env python3
import subprocess


def test1():
    # NOTE: IF we use stdout=subprocess.PIPE,
    # The return result will contain an attribute `stdout`
    # Otherwise, the subprocess prints its output to the console directly
    a = subprocess.run(['ls', '-l'], stdout=subprocess.PIPE)
    assert a.args == ['ls', '-l']
    assert isinstance(a, subprocess.CompletedProcess)
    assert isinstance(a.returncode, int)

    assert isinstance(a.stdout, bytes)
    b = a.stdout.decode()
    assert isinstance(b, str)


def test2():
    # capture_output=True is equivalent to
    # stdout=PIPE, stderr=PIPE
    a = subprocess.run(['ls', '-l'], capture_output=True)
    assert hasattr(a, 'stdout')
    assert hasattr(a, 'stderr')
    assert isinstance(a.stdout, bytes)
    assert isinstance(a.stderr, bytes)
    s = a.stdout.decode()
    assert isinstance(s, str)


def test3():
    a = subprocess.run(['ls', '/abc'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    # NOTE: stderr is redirected to stdout
    s = a.stdout.decode()
    # s is
    # ls: cannot access '/abc': No such file or directory\n


def test4():
    try:
        a = subprocess.run(['ls', '/abc'], capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        #  assert e.args == ['ls', '/abc']
        assert e.cmd == ['ls', '/abc']
        assert e.args == (2, ['ls', '/abc'])
        assert e.returncode != 0
        assert e.output.decode() == ''
        assert 'No such file or directory' in e.stderr.decode()
        # print(e)
        # It prints:
        # Command '['ls', '/abc']' returned non-zero exit status 2.

    a = subprocess.run(['ls', 'abc'],
                       check=False,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    assert a.returncode != 0


def test5():
    # Here, we have to set shell to True, otherwise, the environment
    # variable passed via `env` does not substitue `MYENV`.
    #
    # When shell is True, it is better to pass a string instead of a sequence
    # as args
    a = subprocess.run('ls -lh $MYENV',
                       shell=True,
                       env={'MYENV': './'},
                       stdout=subprocess.DEVNULL)
    #  print(a)


def test6():
    # use cwd to change the current directory
    a = subprocess.run(['ls', '-l', 'passwd'],
                       cwd='/etc',
                       stdout=subprocess.PIPE)
    s = a.stdout.decode()
    assert isinstance(s, str)


def test7():
    a = subprocess.Popen(['ls', '-l'], stdout=subprocess.PIPE)
    assert isinstance(a, subprocess.Popen)

    assert a.poll() is None
    stdout, stderr = a.communicate()  # it will stop until the child finishes
    assert a.poll() == 0

    assert isinstance(stdout, bytes)
    assert stderr is None
    s = stdout.decode()
    assert isinstance(s, str)

    try:
        # NOTE: we cannot invoke its communicate() again
        a.communicate()
    except ValueError:
        # It raises: ValueError: read of closed file
        pass

    # NOTE: in order to get stderr, we have to use subprocess.PIPE here
    a = subprocess.Popen(['ls', '-l'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)

    stdout, stderr = a.communicate()  # it will stop until the child finishes
    assert isinstance(stdout, bytes)
    assert stderr is not None
    s = stdout.decode()
    assert isinstance(s, str)
    assert stderr.decode() == ''

    # If we don't use subprocess.DEVNULL or subprocess.PIPE
    # for stderr, it will print to the console
    a = subprocess.Popen(['ls', '-l', '/abc'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL)
    stdout, stderr = a.communicate()
    assert stdout.decode() == ''
    assert stderr is None


def main():
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()


if __name__ == '__main__':
    main()
