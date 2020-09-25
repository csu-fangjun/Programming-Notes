
Basics
======

attach
------

To attach to a running container:

.. code-block::

  docker exec -it <name> bash

where name can be found by using ``docker container ls | grep <some tag>``.

When enter name, use ``tab`` for auto completion.

history
-------

.. code-block::

  docker history some-tag:latest

set timezone
------------

.. code-block::

    RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
        echo "Asia/Shanghai" > /etc/timezone && \
        \
        apt-get update && apt-get install -y --no-install-recommends \
          tzdata && \
        \
        dpkg-reconfigure -f noninteractive tzdata && \
        \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

.. code-block::

    ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
    echo "Asia/Shanghai" > /etc/timezone
    apt-get update && apt-get install -y --no-install-recommends tzdata
    dpkg-reconfigure -f noninteractive tzdata

expose
------

.. code-block::

    docker run -p <host_port>:<docker_port> -it --rm image:tag

entrypoint
----------

.. code-block::

    docker run --entrypoint /bin/bash -it --rm image:tag
