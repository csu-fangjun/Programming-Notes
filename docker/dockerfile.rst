
Dockerfile
==========



Ubuntu 18.04
------------

.. code-block::

  docker pull ubuntu:18.04

**Save an image to tar file**
  .. code-block::

    docker save --help
    docker save -o ubuntu18.04.tar ubuntu:18.04

**Load an image from a tar file**
  .. code-block::

    docker load --help
    docker load -i ubuntu18.04.tar

**Install Python3**

  .. literalinclude:: ./code/Dockerfile-python-3.6
    :caption: Dockerfile-python-3.6
    :language: dockerfile
    :linenos:


  To install python3.8, use::

    sudo apt-get install python3.8

  To set python3 to point to either python3.8 or python3.6::

    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2

  where:
  - ``/usr/bin/python3`` is a symlink to ``python3`` in ``/etc/alternatives``.
  - the last ``/usr/bin/python3.6`` is the path to the python3 executable
  - ``/etc/alternatives/python3`` is a symlink to ``/usr/bin/python3.6``
  - the last ``1`` and ``2`` is the priority. The larger the number, the higher the priority

Examples
--------

RUN
^^^

.. code-block::

  RUN apt-get update \
      && apt-get install -y --no-install-recommends \
           vim \
           curl \
      && apt-get purge --autoremove -y curl \
      && rm -rf /var/lib/apt/lists/*


ARG
^^^

Refer to `<https://docs.docker.com/engine/reference/builder/#arg>`_.

.. code-block::

  ARG foo
  ARG hello=bar
  RUN echo "foo: ${foo}, hello: ${hello}"

- ``docker build -t some_tag .``, ``${foo}`` is an empty string
- ``docker build -t some_tag --build-arg foo=world .``, ``${foo}`` is the string ``world``
- ``docker build -t some_tag --build-arg hello=world .``, ``${hello}`` is the string ``world``

.. code-block::

  docker build -t some_tag \
    --build-arg foo=world \
    --build-arg hello=Hallo \
    .

ENV
^^^

Refer to `<https://docs.docker.com/engine/reference/builder/#env>`_.

.. code-block::

  ENV hello world
  ENV hallo=Welt
  ENV foo a variable with spaces
  ENV bar="another variable with spaces"

  ENV var1=val1 var2=val2

LABEL
^^^^^

Refer to `<https://docs.docker.com/engine/reference/builder/#label>`.

.. code-block::

  LABEL key1=value1
  LABEL key2=value2 key3=value3

  LABEL key4=value4 \
        key5=value5 \
        key6="with spaces"

  LABEL maintainer="xxx@xxx.com"



Cuda example
------------

- Refer to `<https://github.com/deezer/spleeter/blob/master/docker/cuda-10-0.dockerfile>`_
