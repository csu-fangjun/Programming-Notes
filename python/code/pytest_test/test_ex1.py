import pytest


def inc1(x):
    return x + 1


def test_anwser():
    #  assert inc1(3) == 5
    assert inc1(3) == 4


def f():
    raise SystemExit(1)
    #  pass


def test_f():
    # check that a specified exception is raised
    with pytest.raises(SystemExit):
        f()


# Note that if the function argument contains
# tmpdir, pytest will set its value
# to a temporary directory create by pytest
#
# Run with `pytest -s`
# to see the output of `print`.
#
# pytest --fixtures
#  will show available builtin function arguments
def test_t(tmpdir):
    print('tmpdir', tmpdir)
