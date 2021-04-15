
DDP
===

.. code-block::

  $ ps aux | grep python

  kuangfa+  105774 99.4  0.7 100812136 4096228 pts/21 Rl Apr01 758:04 /root/fangjun/py38/bin/python3 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=6, pipe_handle=8) --multiprocessing-fork


Paper:

  - PyTorch Distributed: Experiences on Accelerating Data Parallel Training

    `<http://www.vldb.org/pvldb/vol13/p3005-li.pdf>`_

Example 1
---------

.. literalinclude:: ./code/ddp/ex1.py
  :caption: code/ddp/ex1.py
  :language: python
  :linenos:

``export CUDA_VISIBLE_DEVICES="1,2,3"``

The above code prints::

  rank: 0
  rank: 1
  rank: 2

Example 2
---------

The following example demonstrates:

  - ``dist.barrier()``
  - ``dist.all_gather()``
  - ``dist.all_reduce()``
  - ``dist.reduce()``
  - ``dist.broadcast()``

.. literalinclude:: ./code/ddp/ex2.py
  :caption: code/ddp/ex2.py
  :language: python
  :linenos:

.. code-block::

  export CUDA_VISIBLE_DEVICES="1,2,3"
  NCCL_ASYNC_ERROR_HANDLING=1 ./ex2.py

prints::

  dist.is_available: True
  rank: 0, 0
  world_size: 3, 3
  rank: 1, 1
  world_size: 3, 3
  rank: 2, 2
  world_size: 3, 3

Example 3
---------

The following example demonstrates the hanging problem
if different nodes have different amount of data.

.. literalinclude:: ./code/ddp/ex3.py
  :caption: code/ddp/ex3.py
  :language: python
  :linenos:

.. code-block::

  export CUDA_VISIBLE_DEVICES="1,2,3"
  NCCL_ASYNC_ERROR_HANDLING=1 ./ex3.py

prints::

  dist.is_available: True
  world_size: 3
  world_size: 3
  world_size: 3
  rank 1 done
  rank 2 done
  Process Process-1:
  Traceback (most recent call last):
    File "/root/fangjun/open-source/pyenv/versions/3.8.6/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
      self.run()
    File "/root/fangjun/open-source/pyenv/versions/3.8.6/lib/python3.8/multiprocessing/process.py", line 108, in run
      self._target(*self._args, **self._kwargs)
    File "/root/fangjun/open-source/.n/pytorch/code/ddp/ex3.py", line 47, in init_process
      fn(rank, world_size)
    File "/root/fangjun/open-source/.n/pytorch/code/ddp/ex3.py", line 28, in run
      y.backward()
    File "/root/fangjun/py38/lib/python3.8/site-packages/torch/tensor.py", line 221, in backward
      torch.autograd.backward(self, gradient, retain_graph, create_graph)
    File "/root/fangjun/py38/lib/python3.8/site-packages/torch/autograd/__init__.py", line 130, in backward
      Variable._execution_engine.run_backward(
  RuntimeError: NCCL communicator was aborted.
