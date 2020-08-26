
Clang
=====


clang language extensions
-------------------------

- `<http://clang.llvm.org/docs/LanguageExtensions.html>`_


clang-format
------------

Create the file ``~/.clang-format`` or place ``.clang-format``
inside the project's root directory. It should contain the following
content::

  BasedOnStyle: Google
  DerivePointerAlignment: false

Then install the vim plugin `google/vim-codefmt <https://github.com/google/vim-codefmt>`_.

``clang-format`` can be installed from pip::

  pip install clang-format

If you do not have ``sudo`` permission, you can get prebuilt clang libraries
from `<https://releases.llvm.org/download.html>`_.

.. WARNING::

  Different versions of ``clang-format`` format the same file differently.
  A specified version of ``clang-format`` should be used for a project.
  Once chosen, the version of ``clang-format`` should be fixed.

Open Source Projects using clang-format
:::::::::::::::::::::::::::::::::::::::

- `<https://github.com/baldurk/renderdoc/wiki/Code-formatting-(using-clang-format)>`_

- `<https://github.com/pytorch/pytorch/blob/master/.clang-format>`_

    PyTorch's ``.clang-format``

- `<https://github.com/Kitware/CMake/blob/master/.clang-format>`_

    CMake's ``.clang-format``

- `<https://github.com/llvm-mirror/llvm/blob/master/.clang-format>`_

    llvm's ``.clang-format``

clang-tidy
----------

If you have ``sudo`` permission, ``clang-tidy`` can be installed by::

  sudo apt-get install clang-tidy

Otherwise, you can download it from
`<https://releases.llvm.org/download.html>`_.

.. HINT::

  It is good practice to always use the latest version of ``clang-tidy``.


The following error:

.. code-block::

    clang-tidy test_clang_tidy.cc
    Error while trying to load a compilation database:
    Could not auto-detect compilation database for file "test_clang_tidy.cc"
    No compilation database found in /xxx/cpp/code/clang or any parent directory
    fixed-compilation-database: Error while opening fixed database: No such file or directory
    json-compilation-database: Error while opening JSON database: No such file or directory
    Running without flags.

is due to the missing file ``compile_commands.json``.

If you are using ``CMake``, then add the following::

.. code-block:: cmake

  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

to ``CMakeLists.txt``. It will generate ``compile_commands.json`` in the build directory automatically.
Copy the file or make a symbolic link to it to the project root directory.

If we want to check only a single file without ``compile_commands.json``, then we can use::

  clang-tidy some_file.cc -- -I<your_include_dir> "and_some_other_compiler_flags"

It documentation is available at `<https://clang.llvm.org/extra/clang-tidy/>`_.


To install it on macOS, use [1]_ ::

  brew install llvm
  ln -s "/usr/local/opt/llvm/bin/clang-format" "/usr/local/bin/clang-format"
  ln -s "/usr/local/opt/llvm/bin/clang-tidy" "/usr/local/bin/clang-tidy"


Open Source Projects using clang-tidy
:::::::::::::::::::::::::::::::::::::::

- `<https://github.com/pytorch/pytorch/blob/master/.clang-tidy>`_

    PyTorch's ``.clang-tidy``

- `<https://github.com/Kitware/CMake/blob/master/.clang-tidy>`_

    CMake's ``.clang-tidy``

- `<https://github.com/llvm-mirror/llvm/blob/master/.clang-tidy>`_

    llvm's ``.clang-tidy``

- `<https://github.com/googleapis/google-cloud-cpp/blob/master/.clang-tidy>`_

    It contains ``CheckOptions``.

clang-check
-----------


.. [1] https://github.com/pisanorg/w/wiki/c-style-tips
