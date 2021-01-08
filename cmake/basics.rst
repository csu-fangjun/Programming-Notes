
Basics
======

CMake 3.1+ is considered as modern CMake.

Bash Completion
---------------

macos
^^^^^

The bash completion file is in
``/usr/local/Cellar/cmake/3.13.2/share/cmake/completions/cmake``

We have to install ``bash-completion`` for macOS first::

  brew install bash-completion
  source /usr/local/etc/bash_completion


Get Help
--------

.. code-block::

  cmake --help-command add_subdirectory
  cmake --help-variable CMAKE_MODULE_PATH
  cmake --help-command-list
  cmake --help-manual-list
  cmake --help-manual cmake-language | less

Useful Variables
----------------

**BUILD_SHARED_LIBS**
  Default value is empty, so ``add_library`` is default to static.

Comment
-------

**bracket argument**
  It has the following form::

    [===[
    any number
    of
    lines
    ]===]

  The number of ``=`` symbols must be the same in the beginning and end.
  The number can be zero.

  Bracket argument is considered as **one** argument. All strings inside it
  are literal. No variable references is evaluated. It is similar to single quote
  in bash.

**block comment**
  ``#`` followed by a bracket argument.

  Example::

    #[=[
    this is
    a multiline
    comment
    ]=]

  Of course, you can put a ``#`` at the begining of every line to acheive the same
  effect.

FetchContent_Declare
====================

For pybind11, The generated scripts are in
``build/_deps/pybind11-subbuild/CMakeFiles/pybind11-populate.dir`` (build.make).
