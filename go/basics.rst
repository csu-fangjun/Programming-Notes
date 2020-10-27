Basics
======

.. literalinclude:: ./code/hello.go
  :caption: hello.go
  :language: go
  :linenos:

The entry function is ``main``. Statements need not be terminated
by ``;``.

There are two methods to run ``hello.go``:
1. ``go run hello.go``. Nothing is generated in the current directory.
2. ``go build hello.go``. It generates a binary ``hello`` in the current directory.
