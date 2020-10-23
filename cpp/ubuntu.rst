
Ubuntu
======

.. code-block::

  docker pull ubuntu:20.04
  docker run -it --rm ubuntu:20.04
  apt-get update
  apt-get install build-essential  # it will install gcc-9
  apt-get install gcc-10
  apt-get install g++-10

  update-alternative --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
  update-alternative --install /usr/bin/gcc gcc /usr/bin/gcc-10 100

``ubuntu:20.10`` installs ``gcc-10`` if ``apt-get install build-essential``
is used.
