
Install
=======

Install from source
-------------------

.. code-block::

  cd open-source/pytorch
  python setup.py build

It will generate ``./build``:

- libraries are in ``./build/lib``
- tests are in ``./build/lib.<os>-<arch>-<python_version>/torch/test``

Add ``./build/lib`` to ``LD_LIBRARY_PATH`` and ``cd`` to the test
directory to run tests.


Some libs
---------

after using ``python setup.py develop``:

- ``caffe2/python/caffe2_pybind11_state.cpython-38-x86_64-linux-gnu.so``
- ``caffe2/python/caffe2_pybind11_state_gpu.cpython-38-x86_64-linux-gnu.so``
- ``torch/lib/libtorch_python.so``
- ``torch/lib/libshm.so``

- ``build/lib.linux-x86_64-3.8``

``convert-caffe2-to-onnx`` is copied to ``miniconda3/envs/py37/bin``

It creats a link from ``envs/py37/lib/python3.8/site-packages/torch.egg-link``
to ``open-source/pytorch``.

nightly wheels
--------------

`<https://download.pytorch.org/whl/nightly/torch_nightly.html>`_
