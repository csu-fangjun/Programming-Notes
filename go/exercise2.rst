Exercise 2
==========

Refer to `<https://golang.org/doc/tutorial/create-module>`_.

.. code-block::

  mkdir greetings
  cd greetings
  go mod init example.com/greetings

It prints::

  go: creating new go.mod: module example.com/greetings

and generates one file ``go.mod``.

.. literalinclude:: ./code/ex2/greetings/go_1.mod
  :caption: go.mod
  :linenos:

Create a new file ``greetings.go``

.. literalinclude:: ./code/ex2/greetings/greetings_1.go-bak
  :caption: greetings.go
  :language: go
  :linenos:

.. code-block::

  cd ..
  mkdir hello
  cd hello

  go mod init example.com/hello

It prints::

  go: creating new go.mod: module example.com/hello

and generates one file ``go.mod``

.. literalinclude:: ./code/ex2/greetings/go_1.mod
  :caption: go.mod
  :linenos:

Create a new file ``hello.go``:

.. literalinclude:: ./code/ex2/hello/hello_1.go-bak
  :caption: hello.go
  :language: go
  :linenos:

Use ``go mod`` to update ``go.mod`` so that we can find
``example.com/greetings``.

.. code-block::

  go mod edit -replace=example.com/greetings=../greetings

``go.mod`` is updated to:

.. literalinclude:: ./code/ex2/greetings/go_2.mod
  :caption: go.mod
  :linenos:

Then run ``go mod tidy`` inside the ``hello`` directory

.. code-block::

  $ go mod tidy
  go: found example.com/greetings in example.com/greetings v0.0.0-00010101000000-000000000000

``go.mod`` is updated to:

.. literalinclude:: ./code/ex2/greetings/go_3.mod
  :caption: go.mod
  :linenos:

.. code-block::

  $ go run .
  Hi, %v. Welcom!Gladys

Now that we can invoke a function from ``greetings`` inside ``hello``.

Now exercises with errors.

.. code-block::

  cd ../greetings

Modify ``greetings.go``.

.. literalinclude:: ./code/ex2/greetings/greetings.go
  :caption: greetings.go
  :language: go
  :linenos:

.. code-block::

  cd ../hello

Update ``hello.go``

.. literalinclude:: ./code/ex2/hello/hello.go
  :caption: hello.go
  :language: go
  :linenos:

.. code-block::

  $ go run .
  greetings: empty name
  exit status 1
