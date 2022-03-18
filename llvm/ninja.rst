ninja
=====

.. code-block:: bash

    pip install ninja

    ninja

will look for the file `$PWD/build.ninja`

.. code-block::

    ninja --help
    ninja -C build -j 20
    ninja -t targets
    ninja -t clean
    ninja -t cleandead

    ninja -v  # be verbose while compiling files

    # suppose hello is a target
    ninja -v hello
    ninja -v -t clean hello

    ninja -n -v hello # dry run
