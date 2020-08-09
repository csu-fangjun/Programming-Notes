
Basics
======

**-c**
  Run simple statement from the bash shell.

  .. code-block:: console

    python3 -c "import sys; print(sys.version)"

    echo "import sys; print(sys.version)" | python3


**-m**
  .. code-block:: console

    python -m foo arg1 arg2       # equivalent to call ./foo.py arg1 arg2
                                  # or if `foo` is a package instead of a module,
                                  # `__main__.py` is executed.

    # if foo is a package, the following is equivalent to `./foo/__main__.py`
    python foo

    python -m http.server 1234    # equivalent to call ./http/server.py 1234

**__name__**
  When a ``.py`` file is executed from the commandline, ``__name__`` is set to ``__main__``;
  when it is imported to another python file, its ``__name__`` is set to the filename withou ``.py``.

  .. code-block:: console

    if __name__ == '__main__':
      pass

**sys.path**
  - The current directory is ``sys.path[0]``
  - ``sys.path[1:]`` is initialized from ``PYTHONPATH``

**sys.argv**
  - ``argv[0]`` is the filename
  - ``argv[1:]`` is the arguments, may be empty

**enumerate**
  enumerate is useful to get the index of every elements
  .. code-block::

    a = list(enumerate(['hello', 'world']))
    assert a == [(0, 'hello'), (1, 'world')]

**hasattr**
  .. code-block::

    a = 'hello'
    assert hasttr(a, '__len__')

**setattr**
  .. code-block::

    class Foo: pass
    f = Foo()
    setattr(f, 'h', 10)
    assert f.h == 10
    assert getattr(f, 'h') == 10

    setattr(f, 'hello world', 100)
    assert getattr(f, 'hello world') == 100

**class**
  .. code-block:: python

    class Foo:
      a = 1
      def __init__(self):
        self.b = 10
      def bar(self): pass
    f = Foo()

  ``Foo.__dict__`` prints::

      {'a': 1, '__module__': '__main__', 'bar': <function bar at 0x7f3ba30f66e0>,
        '__doc__': None, '__init__': <function __init__ at 0x7f3ba30f6668>}

  ``dir(Foo)`` prints::

      ['__doc__', '__init__', '__module__', 'a', 'bar']

  ``f.__dict__`` prints::

    {'b': 10}

  .. NOTE::

    ``f.__dict__`` does not contain the class variable ``a``. When ``f.a = 10`` is
    executed, ``f.__dict__`` is modified and an new key ``a`` is added to ``f.__dict__``;
    ``Foo.__dict__`` is read only!

  ``f.__class__`` prints the same information as ``Foo.__dict__``.

  ``dir(f)`` prints::

      ['__doc__', '__init__', '__module__', 'a', 'b', 'bar']

**__slots__**

**descriptors**
  Combined with ``decorators``, it implements ``staticmethod``, ``classmethod`` and ``property``.

  References:
  - The Inside Story on New-Style Classes

      `<http://python-history.blogspot.com/2010/06/inside-story-on-new-style-classes.html>`_

  - Descriptor HowTo Guide

      `<https://docs.python.org/3/howto/descriptor.html>`_

**abstractmethod**
  Refer to
  - PEP 3119 -- Introducing Abstract Base Classes

      `<https://www.python.org/dev/peps/pep-3119/>`

  and its implementation

    `<https://github.com/python/cpython/blob/3.8/Lib/_py_abc.py>`_

    `<https://bugs.python.org/issue1706989>`_

  It the attribute ``__abstractmethods__`` is not empty, it is implemented
  in c that the class cannot be instantiated.

whl
---

To view the dependencies of a whl file, use

.. code-block::

  pip install pkginfo
  pkginfo -f requires_dist /path/to.whl

The cache of ``pip`` saves the installed whl files.
The path is ``$HOME/.cache/pip``.

To install librosa:

.. code-block::

  sudo apt-get install llvm-8
  # add a symlink llvm-config, which links to llvm-config-8
  pip install llvmlite
