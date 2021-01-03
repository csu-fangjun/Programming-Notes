Address Sanitizer
=================

Refer to `<https://github.com/google/sanitizers/wiki/AddressSanitizer>`_

Use after free
--------------

.. literalinclude:: ./code/asan/ex1.c
  :caption: use after free (code/asan/ex1.c)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex1.c -o ex1
  ./ex1 &> ex1_out.txt

.. literalinclude:: ./code/asan/ex1_out.txt
  :caption: Output of use after free (code/asan/ex1_out.txt)

Double free (delete)
--------------------

.. literalinclude:: ./code/asan/ex2.cc
  :caption: double free (delete) (code/asan/ex2.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex2.c -o ex2
  ./ex2 &> ex2_out.txt

.. literalinclude:: ./code/asan/ex2_out.txt
  :caption: Output of double free (delete) (code/asan/ex2_out.txt)

malloc + delete
---------------

.. literalinclude:: ./code/asan/ex3.cc
  :caption: malloc + delete (code/asan/ex3.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex3.c -o ex3
  ./ex3 &> ex3_out.txt

.. literalinclude:: ./code/asan/ex3_out.txt
  :caption: Output of malloc + delete (code/asan/ex3_out.txt)

new + free
----------

.. literalinclude:: ./code/asan/ex4.cc
  :caption: new + free (code/asan/ex4.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex4.c -o ex4
  ./ex4 &> ex4_out.txt

.. literalinclude:: ./code/asan/ex4_out.txt
  :caption: Output of new + free (code/asan/ex4_out.txt)

new[] + delete
--------------

.. literalinclude:: ./code/asan/ex5.cc
  :caption: new[] + delete (code/asan/ex5.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex5.c -o ex5
  ./ex5 &> ex5_out.txt

.. literalinclude:: ./code/asan/ex5_out.txt
  :caption: Output of new[] + delete (code/asan/ex5_out.txt)

new + delete[]
--------------

.. literalinclude:: ./code/asan/ex6.cc
  :caption: new[] + delete (code/asan/ex6.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex6.c -o ex6
  ./ex6 &> ex6_out.txt

.. literalinclude:: ./code/asan/ex6_out.txt
  :caption: Output of new + delete[] (code/asan/ex6_out.txt)

new[] + std::unique_ptr<>
-------------------------

.. literalinclude:: ./code/asan/ex7.cc
  :caption: new[] + delete (code/asan/ex7.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex7.c -o ex7
  ./ex7 &> ex7_out.txt

.. literalinclude:: ./code/asan/ex7_out.txt
  :caption: Output of new + delete[] (code/asan/ex7_out.txt)

Memory leak (new)
-----------------

.. literalinclude:: ./code/asan/ex8.cc
  :caption: Memory leak (new) (code/asan/ex8.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex8.c -o ex8
  ./ex8 &> ex8_out.txt

.. literalinclude:: ./code/asan/ex8_out.txt
  :caption: Output of memory leak (new) (code/asan/ex8_out.txt)

Buffer overflow (new)
---------------------

.. literalinclude:: ./code/asan/ex9.cc
  :caption: Buffer overflow (new) (code/asan/ex9.cc)
  :language: cpp
  :linenos:

.. code-block::

  gcc -fsanitize=address -O1 -fno-omit-frame-pointer -g ex9.c -o ex9
  ./ex9 &> ex9_out.txt

.. literalinclude:: ./code/asan/ex9_out.txt
  :caption: Buffer overflow (new) (code/asan/ex9_out.txt)
