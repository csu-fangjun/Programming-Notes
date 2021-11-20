Installation
============

.. code-block:: bash

  ./cuda_10.1.243_418.87.00_linux.run \
    --silent \
    --extract=/root/fangjun/software/cuda-10.1.243

Install cuda 11.3.0:

.. code-block::

  ./cuda_11.3.0_465.19.01_linux.run \
    --silent \
    --toolkit \
    --samples \
    --no-drm \
    --no-man-page \
    --no-opengl-libs \
    --installpath=/ceph-fj/fangjun/software/cuda-11.3.0



To install cudnn,

.. code-block::

  cd software/
  # there is software/cuda
  tar xvf /path/to/cudnn.tar.gz

or

.. code-block::

  cd software/cuda-11.1
  tar xvf /path/to/cudnn.tar.gz --strip-components=1
