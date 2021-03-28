Exercise 1
==========

Refer to `<https://golang.org/doc/tutorial/getting-started>`_

The exercise shows:

  1. How to print ``Hello, World`` in ``go``
  2. How to external packages from ``pkg.go.dev``.
  3. How to use ``go mod init`` and ``go mod tidy``


.. code-block::

  mkdir hello
  cd hello

  go mod init example.com/hello

It prints::

  go: creating new go.mod: module example.com/hello

A new file ``go.mod`` is generated.

.. literalinclude:: ./code/ex1/hello/go_1.mod
  :caption: go.mod
  :linenos:


Then create a file ``hello.go``

.. literalinclude:: ./code/ex1/hello/hello_1.go-bak
  :caption: hello.go
  :language: go
  :linenos:

Then run the code:

.. code-block::

  go run .

And it prints::

  Hello, World

Visit `<https://pkg.go.dev/search?q=quote>`_ to find the package ``rsc.io/quote``,
which is located at `<https://pkg.go.dev/rsc.io/quote>`_.

It source file is located at `<https://github.com/rsc/quote/blob/v1.5.2/quote.go>`_.

Then modify ``hello.go`` to:

.. literalinclude:: ./code/ex1/hello/hello.go
  :caption: hello.go
  :language: go
  :linenos:

Then run ``go mod tidy``:

.. code-block::

  $ go mod tidy
  go: finding module for package rsc.io/quote
  go: downloading rsc.io/quote v1.5.2
  go: found rsc.io/quote in rsc.io/quote v1.5.2
  go: downloading rsc.io/sampler v1.3.0
  go: downloading golang.org/x/text v0.0.0-20170915032832-14c0d48ead0c

It will download some files to ``~/go/`` and one file ``go.sum`` in the current directory.

.. literalinclude:: ./code/ex1/hello/go.sum
  :caption: go.sum
  :linenos:

.. code-block::

  $ ls -l ~/go/pkg/mod

  total 12
  drwxr-xr-x 3 kuangfangjun root 4096 Mar 28 21:46 cache
  drwxr-xr-x 3 kuangfangjun root 4096 Mar 28 21:46 golang.org
  drwxr-xr-x 4 kuangfangjun root 4096 Mar 28 21:46 rsc.io

  $ ls -l ~/go/pkg/mod/rsc.io/

  total 8
  dr-xr-xr-x 3 kuangfangjun root 4096 Mar 28 21:46 quote@v1.5.2
  dr-xr-xr-x 2 kuangfangjun root 4096 Mar 28 21:46 sampler@v1.3.0

  $ ls -l ~/go/pkg/mod/rsc.io/*

  /root/fangjun/go/pkg/mod/rsc.io/quote@v1.5.2:
  total 24
  -r--r--r-- 1 kuangfangjun root 1479 Mar 28 21:46 LICENSE
  -r--r--r-- 1 kuangfangjun root  131 Mar 28 21:46 README.md
  dr-xr-xr-x 2 kuangfangjun root 4096 Mar 28 21:46 buggy
  -r--r--r-- 1 kuangfangjun root   55 Mar 28 21:46 go.mod
  -r--r--r-- 1 kuangfangjun root  793 Mar 28 21:46 quote.go
  -r--r--r-- 1 kuangfangjun root  917 Mar 28 21:46 quote_test.go

  /root/fangjun/go/pkg/mod/rsc.io/sampler@v1.3.0:
  total 40
  -r--r--r-- 1 kuangfangjun root  1479 Mar 28 21:46 LICENSE
  -r--r--r-- 1 kuangfangjun root 12820 Mar 28 21:46 glass.go
  -r--r--r-- 1 kuangfangjun root   729 Mar 28 21:46 glass_test.go
  -r--r--r-- 1 kuangfangjun root    88 Mar 28 21:46 go.mod
  -r--r--r-- 1 kuangfangjun root  3840 Mar 28 21:46 hello.go
  -r--r--r-- 1 kuangfangjun root   672 Mar 28 21:46 hello_test.go
  -r--r--r-- 1 kuangfangjun root  2048 Mar 28 21:46 sampler.go



``go run .`` prints::

  Don't communicate by sharing memory, share memory by communicating.


``~/go/pkg/mod/rsc.io/quote@v1.5.2/go.mod``:

.. code-block::
  :caption: rsc.io/quote/go.mod

  module "rsc.io/quote"

  require "rsc.io/sampler" v1.3.0

``~/go/pkg/mod/rsc.io/quote@v1.5.2/quote.go``:

.. code-block:: go
  :caption: rsc.io/quote/quote.go

  // Copyright 2018 The Go Authors. All rights reserved.
  // Use of this source code is governed by a BSD-style
  // license that can be found in the LICENSE file.

  // Package quote collects pithy sayings.
  package quote // import "rsc.io/quote"

  import "rsc.io/sampler"

  // Hello returns a greeting.
  func Hello() string {
    return sampler.Hello()
  }

  // Glass returns a useful phrase for world travelers.
  func Glass() string {
    // See http://www.oocities.org/nodotus/hbglass.html.
    return "I can eat glass and it doesn't hurt me."
  }

  // Go returns a Go proverb.
  func Go() string {
    return "Don't communicate by sharing memory, share memory by communicating."
  }

  // Opt returns an optimization truth.
  func Opt() string {
    // Wisdom from ken.
    return "If a program is too slow, it must have a loop."
  }
