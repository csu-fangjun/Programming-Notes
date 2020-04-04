
Basics
======

Get the current directory of the script
---------------------------------------

.. code-block:: bash

  cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
  echo $cur_dir

``date``
--------

.. code-block:: bash

  date "+%F %T"
  # 2010-01-02 08:01:02

``array``
---------

.. code-block:: bash

  a=(
  hello
  world
  )

  for i in ${a[@]}; do
    echo $i
  done

Output

.. code-block:: console

  hello
  world

Block Comment
-------------

Refer to `Block Comments in a Shell Script`_.

.. code-block:: bash

  :<<'EOF'
  hello'world
  foo
  bar
  EOF

.. HINT::

  ``:`` can be omitted. But it is not portable if ``:`` is missing.


.. _Block Comments in a Shell Script: https://stackoverflow.com/questions/947897/block-comments-in-a-shell-script/947936#947936
