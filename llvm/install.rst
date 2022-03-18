Installation
============

Refer to `<https://llvm.org/docs/GettingStarted.html#requirements>`_

.. cod-block:: bash

    git clone --depth 1 https://github.com/llvm/llvm-project.git
    cd llvm-project
    mkdir build
    cd build
    cmake -G Ninja ../llvm
    cmake --build . --target <target>
    # or
