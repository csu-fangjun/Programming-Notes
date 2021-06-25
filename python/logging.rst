Logging
=======

Refer to

  - Basic Logging Tutorial `<https://docs.python.org/3/howto/logging.html#logging-basic-tutorial>`_

  - `<https://docs.python.org/3/howto/logging-cookbook.html#logging-cookbook>`)


Example 1
---------

.. literalinclude:: ./code/logging_test/ex1.py
  :caption: code/logging_test/ex1.py
  :language: python
  :linenos:

The output of the above code is::

  WARNING:root:watch out!

.. note::

  The default logging level is ``WARN``, so ``logging.info`` is not printed.

Example 2
---------

.. literalinclude:: ./code/logging_test/ex2.py
  :caption: code/logging_test/ex2.py
  :language: python
  :linenos:

The above code logs to a file ``ex2.log`` in the current directory.

.. caution::

  ``ex2.log`` is created if it does not exist. If it exists, it uses
  ``append`` mode to open it!

  ``logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)``
  can be used to delete previous logs.


  The above code does not output to the console!

It prints the following to the file ``ex2.log``::

  DEBUG:root:This message should go to the log file
  INFO:root:So should this
  WARNING:root:And this too
  INFO:root:Also this

Example 3
---------

.. code-block::

  import logging
  numeric_level = getattr(logging, 'DEBUG') # numeric_level is 10
  assert isinstance(numeric_level, int)

  assert logging.DEBUG == 10

  logging.basicConfig(level=numeric_level)

See `<https://docs.python.org/3/library/logging.html#levels>`_
for a list of log levels.

Example 4
---------

.. literalinclude:: ./code/logging_test/ex4.py
  :caption: code/logging_test/ex4.py
  :language: python
  :linenos:

.. CAUTION::

  The above code prints nothing to the console.
  ``logging.basicConfig`` should be called before any calls
  to ``logging.xxx()``.

Example 5
---------

.. literalinclude:: ./code/logging_test/ex5.py
  :caption: code/logging_test/ex5.py
  :language: python
  :linenos:

The above code prints the following to the console::

  INFO:root:output
  INFO:root:also output

Example 6
---------

Use formatted messages.

.. literalinclude:: ./code/logging_test/ex6.py
  :caption: code/logging_test/ex6.py
  :language: python
  :linenos:

The above code prints the following to the console::

  DEBUG:debug
  INFO:info
  WARNING:warning

Example 7
---------

Use formatted messages with date and time.

.. literalinclude:: ./code/logging_test/ex7.py
  :caption: code/logging_test/ex7.py
  :language: python
  :linenos:

The above code prints the following to the console::

  2021-04-12 11:13:19,238 DEBUG:debug
  2021-04-12 11:13:19,238 INFO:info
  2021-04-12 11:13:19,238 WARNING:warning


Example 8
---------

Use formatted messages with formatted date and time.

See `<https://docs.python.org/3/library/time.html#time.strftime>`_
for available format flag for time.

.. literalinclude:: ./code/logging_test/ex8.py
  :caption: code/logging_test/ex8.py
  :language: python
  :linenos:

The above code prints the following to the console::

  2021-04-12 11:18:52 DEBUG:debug
  2021-04-12 11:18:52 INFO:info
  2021-04-12 11:18:52 WARNING:warning


TODOs
-----

- `<https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/>`_

    SpeechBrain is using this kind of techniques.

    See `<https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/utils/logger.py>`_
