
Pip
===

.. code-block:: bash

   pip install  -e . # Install the current project in edit mode

Package version
---------------

- `PEP 440 -- Version Identification and Dependency Specification <https://www.python.org/dev/peps/pep-0440/>`_

Example:
- ``major.minor`` or  ``major.minor.micro``
- ``0.9``
- ``0.9.1``
- ``0.9.2``
- ``0.9.3``
- ``0.9.10``
- ``0.9.11``
- ``1.0``
- ``1.0.1``
- ``1.1``
- ``2.0``
- ``2.0.1``

Filename Convention
-------------------

- `PEP 491 -- The Wheel Binary Package Format 1.9 <https://www.python.org/dev/peps/pep-0491/>`_

- `PEP 513 -- A Platform Tag for Portable Linux Built Distributions <https://www.python.org/dev/peps/pep-0513/>`_

    Describe manylinux. It introduces two platform tags: ``manylinux1_x86_64`` and ``manylinux_x86``.

- `PEP 571 -- The manylinux2010 Platform Tag <https://www.python.org/dev/peps/pep-0571/>`_

    It defines another platform tag: ``manylinux2010``

- `PEP 427 -- The Wheel Binary Package Format 1.0 <https://www.python.org/dev/peps/pep-0427/>`_

    Describe what are inside of `*.whl`.

- `PEP 376 -- Database of Installed Python Distributions <https://www.python.org/dev/peps/pep-0376/>`_

    Specify the content in the folder `xxx.dist-info`

- `PEP 425 -- Compatibility Tags for Built Distributions <https://www.python.org/dev/peps/pep-0425/>`_

    It defines `python_tag - abi_tag - platform_tag`.

    Python tag: py27, cp33. ``py`` means generic python; ``cp`` means CPython.

    abi tag: `cp32dmu`, none

    platform tag: any, linux_x86_64, linux_x86, win32

manylinux
---------

- numpy uses `docker pull quay.io/pypa/manylinux2010_i686 <https://github.com/numpy/numpy/blob/master/azure-pipelines.yml#L37>`_

- MegEngine: `<https://github.com/MegEngine/MegEngine/blob/master/scripts/whl/manylinux2010/Dockerfile>`_

  .. code-block:

    FROM quay.io/pypa/manylinux2010_x86_64:2020-01-31-046f791

