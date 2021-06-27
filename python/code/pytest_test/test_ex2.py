# test fixtures

# see https://docs.pytest.org/en/6.2.x/fixture.html

import pytest


@pytest.fixture
def foo():
    print('called foo')
    return 3


# Note that the argument of test_case1 is foo,
# which is also a function name that decorated
# by @pytest.fixture
#
# Run it with pytest -s test_ex2.py
def test_case1(foo):
    print('foo is', foo)  # foo is 3


@pytest.fixture
def bar():
    return 5


@pytest.fixture
def my_bar(bar):
    return [bar, 6]


# Note that it depends on two fixtures
def test_case2(my_bar, foo):
    print('my_bar', my_bar)  # my_bar [5, 6]
    print('foo', foo)  # foo 3
