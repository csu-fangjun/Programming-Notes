
Build with setup.py
===================

To compile the following code,

.. literalinclude:: ./code/hello.cc
  :caption: hello.cc
  :language: cpp
  :linenos:

We can use

.. literalinclude:: ./code/setup.py
  :caption: setup.py
  :language: python
  :linenos:

Execute ``python setup.py build`` and it will generate
``build/lib.linux-x86_64-3.5/hello.cpython-35m-x86_64-linux-gnu.so``.


.. code-block:: bash

    >>> import sys
    >>> sys.path.insert(0, 'build/lib.linux-x86_64-3.5')
    >>> import hello
    >>> hello.add(1, 2)
    3

To install the package, use

.. code-block:: bash

    python setup.py build

which shows::

    running install
    running bdist_egg
    running egg_info
    writing dependency_links to hello_world_package.egg-info/dependency_links.txt
    writing hello_world_package.egg-info/PKG-INFO
    writing top-level names to hello_world_package.egg-info/top_level.txt
    reading manifest file 'hello_world_package.egg-info/SOURCES.txt'
    writing manifest file 'hello_world_package.egg-info/SOURCES.txt'
    installing library code to build/bdist.linux-x86_64/egg
    running install_lib
    running build_ext
    creating build/bdist.linux-x86_64/egg
    copying build/lib.linux-x86_64-3.5/hello.cpython-35m-x86_64-linux-gnu.so -> build/bdist.linux-x86_64/egg
    creating stub loader for hello.cpython-35m-x86_64-linux-gnu.so
    byte-compiling build/bdist.linux-x86_64/egg/hello.py to hello.cpython-35.pyc
    creating build/bdist.linux-x86_64/egg/EGG-INFO
    copying hello_world_package.egg-info/PKG-INFO -> build/bdist.linux-x86_64/egg/EGG-INFO
    copying hello_world_package.egg-info/SOURCES.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
    copying hello_world_package.egg-info/dependency_links.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
    copying hello_world_package.egg-info/top_level.txt -> build/bdist.linux-x86_64/egg/EGG-INFO
    writing build/bdist.linux-x86_64/egg/EGG-INFO/native_libs.txt
    zip_safe flag not set; analyzing archive contents...
    __pycache__.hello.cpython-35: module references __file__
    creating 'dist/hello_world_package-1.0-py3.5-linux-x86_64.egg' and adding 'build/bdist.linux-x86_64/egg' to it
    removing 'build/bdist.linux-x86_64/egg' (and everything under it)
    Processing hello_world_package-1.0-py3.5-linux-x86_64.egg
    creating /path/to/py35/lib/python3.5/site-packages/hello_world_package-1.0-py3.5-linux-x86_64.egg
    Extracting hello_world_package-1.0-py3.5-linux-x86_64.egg to /path/to/py35/lib/python3.5/site-packages
    Adding hello-world-package 1.0 to easy-install.pth file

    Installed /path/to/py35/lib/python3.5/site-packages/hello_world_package-1.0-py3.5-linux-x86_64.egg
    Processing dependencies for hello-world-package==1.0
    Finished processing dependencies for hello-world-package==1.0

To list the information of the installed package, use

.. code-block:: bash

    pip show --files hello-world-package

which should print::

    Name: hello-world-package
    Version: 1.0
    Summary: hello world in pybind11
    Home-page: https://github.com/csu-fangjun
    Author: fangjun
    Author-email: fangjun dot kuang at gmail dot com
    License: UNKNOWN
    Location: /path/to/py35/lib/python3.5/site-packages/hello_world_package-1.0-py3.5-linux-x86_64.egg
    Requires:
    Required-by:
    Files:
    Cannot locate installed-files.txt

To uninstall the package, use

.. code-block:: bash

    pip uninstall hello-world-package

It prints::

    Found existing installation: hello-world-package 1.0
    Uninstalling hello-world-package-1.0:
      Would remove:
          /path/to/py35/lib/python3.5/site-packages/hello_world_package-1.0-py3.5-linux-x86_64.egg
      Proceed (y/n)? y
        Successfully uninstalled hello-world-package-1.0
