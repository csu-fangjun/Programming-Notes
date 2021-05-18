Installation
============

.. code-block:: bash

  ./cuda_10.1.243_418.87.00_linux.run \
    --silent \
    --extract=/root/fangjun/software/cuda-10.1.243



To install cudnn,

.. code-block::

  cd software/
  # there is software/cuda
  tar xvf /path/to/cudnn.tar.gz

or

.. code-block::

  cd software/cuda-11.1
  tar xvf /path/to/cudnn.tar.gz --strip-components=1
