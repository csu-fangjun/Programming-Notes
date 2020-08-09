
.. toctree::
  :maxdepth: 3

Makefile
========

To run a file named ``info.mk``, use ``make -f info.mk``.

Automatic Variables
-------------------

Refer to `<https://www.gnu.org/software/make/manual/html_node/Automatic-Variables.html#Automatic-Variables>`_

- ``$@``, the filename of the output target
- ``$<``, the name of the **first** prerequisite
- ``$^``, the name of **all** prerequisites

Conditional
-----------

Refer to `<https://www.gnu.org/software/make/manual/html_node/Conditionals.html#Conditionals>`_.

Define
------

Refer to `<https://www.gnu.org/software/make/manual/html_node/Canned-Recipes.html#Canned-Recipes>`_
and `<https://www.gnu.org/software/make/manual/html_node/Multi_002dLine.html#Multi_002dLine>`_.

.. literalinclude:: ./code/makefile_notes/define.mk
  :caption: define.mk
  :language: makefile
  :linenos:

Functions
---------

Refer to `<https://www.gnu.org/software/make/manual/html_node/Functions.html#Functions>`_.

A hello world example:

.. code-block::

  $(info "hello world")
  all:

Another hello world example:

.. code-block::

  s := "hello world"
  $(info $(s))
  all:

Text functions
--------------

Refer to `<https://www.gnu.org/software/make/manual/html_node/Text-Functions.html#Text-Functions>`_

subst
^^^^^

``$(subst from,to,text)``

.. literalinclude:: ./code/makefile_notes/subst.mk
  :caption: subst.mk
  :language: makefile
  :linenos:

patsubst
^^^^^^^^

pattern subst.

``$(patsubst pattern,replacement,text)``

.. note::

  ``$(var:suffix=replacement)`` is equivalent to ``$(patsubst %suffix,%replacement,$(var))``

.. literalinclude:: ./code/makefile_notes/patsubst.mk
  :caption: patsubst.mk
  :language: makefile
  :linenos:


strip
^^^^^

``$(strip string)`` removes leading and trailing spaces. Reduces mutliple contiguous internal spaces to one space.

.. literalinclude:: ./code/makefile_notes/strip.mk
  :caption: strip.mk
  :language: makefile
  :linenos:

findstring
^^^^^^^^^^

``$(findstring substr,string)`` returns substr if it is in string, else return empty.

.. literalinclude:: ./code/makefile_notes/findstring.mk
  :caption: findstring.mk
  :language: makefile
  :linenos:

filter
^^^^^^

- ``$(filter pattern,$(str))``
- ``$(filter pattern1 pattern2,$(str))``
- ``$(filter pattern1 pattern2 pattern3,$(str))``

Return a list of whitespace separated words that match the given pattern or patterns.
``%`` is used for a pattern.

.. literalinclude:: ./code/makefile_notes/filter.mk
  :caption: filter.mk
  :language: makefile
  :linenos:

filter-out
^^^^^^^^^^

It is the opposite of ``filter``, which removes any words that match the pattern and
keep those words that do NOT match the pattern.

.. literalinclude:: ./code/makefile_notes/filter-out.mk
  :caption: filter-out.mk
  :language: makefile
  :linenos:

sort
^^^^

``$(sort $(str))``, sort and unique.

.. literalinclude:: ./code/makefile_notes/sort.mk
  :caption: sort.mk
  :language: makefile
  :linenos:

word
^^^^

``$(word n,$(text))`` returns the ``n-th`` word of the text.
``n`` starts from 1. If ``n`` is greater than the number of available words, then
it returns empty.

.. literalinclude:: ./code/makefile_notes/word.mk
  :caption: word.mk
  :language: makefile
  :linenos:

wordlist
^^^^^^^^

``$(wordlist start,end,$(text))``

- ``start`` counts from 1
- ``start`` and ``end`` are both inclusive
- if ``end`` is less than ``start``, return empty
- if ``end`` is greater than the number of available words, it is the number of available words
- if ``start`` is greater than the number of available words, return empty

.. literalinclude:: ./code/makefile_notes/wordlist.mk
  :caption: wordlist.mk
  :language: makefile
  :linenos:

words
^^^^^

``$(words $(text))``, return the number of words in ``text``.

.. literalinclude:: ./code/makefile_notes/words.mk
  :caption: words.mk
  :language: makefile
  :linenos:

firstword
^^^^^^^^^

``$(firstword $(text))``, return the first word in ``text``.

Compared with ``$(word 1,$(text))``, ``firstword`` is simpler.

.. literalinclude:: ./code/makefile_notes/firstword.mk
  :caption: firstword.mk
  :language: makefile
  :linenos:

lastword
^^^^^^^^

``$(lastword $(text))``, return the last word in ``text``.

Compared with ``$(word $(words $(text)),$(text))``, ``lastword`` is simpler.

.. literalinclude:: ./code/makefile_notes/lastword.mk
  :caption: lastword.mk
  :language: makefile
  :linenos:

Filename functions
------------------

Refer to `<https://www.gnu.org/software/make/manual/html_node/File-Name-Functions.html#File-Name-Functions>`_.

dir
^^^

``$(dir $(text))``

.. literalinclude:: ./code/makefile_notes/dir.mk
  :caption: dir.mk
  :language: makefile
  :linenos:


notdir
^^^^^^

``$(notdir $(text))``

.. literalinclude:: ./code/makefile_notes/notdir.mk
  :caption: notdir.mk
  :language: makefile
  :linenos:

suffix
^^^^^^

``$(suffix $(text))``

.. literalinclude:: ./code/makefile_notes/suffix.mk
  :caption: suffix.mk
  :language: makefile
  :linenos:


References
----------

- `<https://www.gnu.org/software/make/manual/html_node/index.html>`_

    Manual
