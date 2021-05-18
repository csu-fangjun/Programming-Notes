
Regular Expression
==================

Basics
------

- ``\w`` is equivalent to ``[a-zA-Z0-9_]``
- ``\W``, ``[^a-zA-Z0-9_]``
- ``\d`` is equivalent to ``[0-9]``
- ``\D``, ``[^0-9]``
- ``\s``, ``[ \t\n\r\f\v]``
- ``\S``, ``[^ \t\n\r\f\v]``
- ``[\s,.]`` matches any whitespace character, or the comma ``,``, or the dot ``.``
- zero or more -> ``*``
- one or more -> ``+``
- zero or one -> ``?``


Examples
--------

- decimal numerals in c:

.. code-block::

    0|[1-9][0-9]*

- all numerals in c

.. code-block::

  [1-9][0-9]*|0[xX][0-9a-fA-F]+|0[0-7]*

- floating point numerals

.. code-block::

  1.2
  1.
  0.1
  .1
  1e-2

  (\d+\.\d*|\d*\.\d+)([eE][-+]?\d+)?

- identifiers in c

.. code-block::

  [a-zA-Z_][a-zA-Z0-9_]*

- comments in c++

.. code-block::

  // xxxx
  /* xxx */
  we must exclude /* */   ***/

  //.*|/\*([^*]|\*[^/])*\*+/


Extensions
----------

See `<https://docs.python.org/3/library/re.html>`_

It has the form ``(?...)``


References
----------

- Regular Expression HOWTO

    `<https://docs.python.org/3/howto/regex.html#regex-howto>`_


