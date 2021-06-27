import pytest


@pytest.fixture(scope='function', autouse=True)
def foo():
    print('test started')
    yield
    print('test ended')


def test_case():
    print('in test')


# pytest -s ./test_ex3.py
# prints
# test started
# in test
# test ended
