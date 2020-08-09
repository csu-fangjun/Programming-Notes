
AWK
===

**Code Examples**
  .. code-block::

    BEGIN { print "START" }
          { print         }
    END   { print "STOP"  }

  .. code-block::

    BEGIN { print "File\tOwner"}
          { print $8, "\t", $3}
    END   { print " - DONE -" }

  Note that the first field is ``$1``. It counts from ``0``. ``print`` is used for ``printing``.
  The comma ``,`` is not shown and is converted to a space in the output. Remember to use a single
  quote in the terminal to use ``$1``; otherwise, ``$1`` is substitued by bash.

  ``;`` is used to separate statements in the same line. If there is only one statement per line,
  then there is no need to use ``;``.

  .. code-block::

    ls -l | awk '{print $9, $3}'
    ls -l | awk '{printf("%s %s\n",$9, $3)}'

    ls -l | awk '
    {print $9, $3}
    '

  Note that ``''`` can span multiple lines. It is useful for readability.

  .. code-block::

    echo "1 3" | awk 'print $1/$2'

  It prints ``0.33333``. Note that it uses ``float``.

  .. code-block::

    (echo "hello"; echo "world"; echo "foo") | awk 'match($1, /oo/) {print $1}'  # print foo
    (echo "hello"; echo "world"; echo "foo") | awk 'match($1, /ll/) {print $1}'  # print hello
    (echo "hello"; echo "world"; echo "foo") | awk 'match($1, /l/) {print $1}'   # print hello, world

    # the following matches a whole line
    (echo "hello"; echo "world"; echo "foo") | awk '/oo/ {print $1}'  # print foo
    (echo "hello"; echo "world"; echo "foo") | awk '/ll/ {print $1}'  # print hello
    (echo "hello"; echo "world"; echo "foo") | awk '/l/ {print $1}'   # print hello, world

  Inside ``/ /`` is a regular expression. 

  .. code-block::

    echo "1;2;3" | sed "s/;/\n/g" | awk 'BEGIN{sum = 0}; {sum += $1}; END{print "sum is", sum}'

  Note that we can use the same ``+=`` as C language. The variable ``sum`` is defined in ``BEGIN``.

  .. code-block::

    echo "1;2;3" | sed "s/;/\n/g" | awk -v sum=10 '{sum += $1}; END{print "sum is", sum}'

  We pass a variable ``sum`` to awk with an initial value 10. It prints 16.

  ``NF`` is a built-in variable, indicating number of fields of the current line. Note that
  it is ``NF``, NOT ``$NF``.

  .. code-block::

    ls -l | awk 'print $NF' # print the last field


  ``NR`` is the number of lines process sor far. It counts from 1.


References
----------

- Intro to AWK `<https://www.grymoire.com/Unix/Awk.html>`_
