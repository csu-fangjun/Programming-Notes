
Cereal
======

- GitHub address: `<https://github.com/USCiLab/cereal>`_

- Documentation address: `<http://uscilab.github.io/cereal/>`_

- Doxygen API address: `<https://uscilab.github.io/cereal/assets/doxygen/index.html>`_

- Quick Start: `<http://uscilab.github.io/cereal/quickstart.html>`_



Installation
------------

It is a header only library.

.. code-block:: bash

  wget https://github.com/USCiLab/cereal/archive/v1.3.0.tar.gz
  tar xf v1.3.0.tar.gz

Now all stuff related to cereal is in the folder ``cereal-1.3.0``

Hello World Example
-------------------

.. literalinclude:: ./code/cereal/Makefile
   :caption: Makefile
   :language: makefile
   :linenos:

.. literalinclude:: ./code/cereal/hello.cc
   :caption: hello.cc
   :language: cpp
   :lineno-start: 4
   :lines: 4-
   :linenos:

Output
::::::

.. code-block:: console

  {
      "value0": {
          "value0": 2020,
          "value1": [
              {
                  "value0": 1,
                  "value1": "s1",
                  "value2": [
                      1,
                      2,
                      3
                  ]
              },
              {
                  "value0": 2,
                  "value1": "s2",
                  "value2": [
                      10,
                      20
                  ]
              }
          ]
      }
  }<?xml version="1.0" encoding="utf-8"?>
  <cereal>
    <value0>
      <value0>2020</value0>
      <value1 size="dynamic">
        <value0>
          <value0>1</value0>
          <value1>s1</value1>
          <value2 size="dynamic">
            <value0>1</value0>
            <value1>2</value1>
            <value2>3</value2>
          </value2>
        </value0>
        <value1>
          <value0>2</value0>
          <value1>s2</value1>
          <value2 size="dynamic">
            <value0>10</value0>
            <value1>20</value1>
          </value2>
        </value1>
      </value1>
    </value0>
  </cereal>



Explanation
:::::::::::

.. literalinclude:: ./code/cereal/hello.cc
   :language: cpp
   :lineno-start: 8
   :lines: 8-10
   :linenos:

imports the necessary headers. Cereal supports three formats:

  - binary
  - json
  - xml

.. WARNING::

  According to `<http://uscilab.github.io/cereal/serialization_archives.html>`_,
  binary data format ignores endianess.

  Use ``"cereal/archives/portable_binary.hpp"`` to support both big and little endianess.

.. literalinclude:: ./code/cereal/hello.cc
   :language: cpp
   :lineno-start: 11
   :lines: 11-12
   :linenos:

imports support for standard library types.

.. HINT::

  Refer to `<https://uscilab.github.io/cereal/stl_support.html>`_ for more STL types.

.. literalinclude:: ./code/cereal/hello.cc
   :language: cpp
   :lineno-start: 21
   :lines: 21-24
   :linenos:

defines ``serialize``, which is used for both reading and writing.

.. HINT::

  ``ar(id, name, courses);`` can be writen as::

      ar(id); ar(name); ar(courses);

  Or written as::

      ar(id); ar(name, courses);


.. literalinclude:: ./code/cereal/hello.cc
   :language: cpp
   :lineno-start: 30
   :lines: 30-47
   :linenos:

shows two alternatives to add support for cereal to a customized type:

  - define ``load`` and ``save``
  - **or** define ``serialize``

If both ``load``, ``save`` and ``serialize`` are defined, it results
in a compile error.


Cereal Internals
----------------
