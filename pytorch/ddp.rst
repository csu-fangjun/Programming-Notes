
DDP
===

.. code-block::

  $ ps aux | grep python

  kuangfa+  105774 99.4  0.7 100812136 4096228 pts/21 Rl Apr01 758:04 /root/fangjun/py38/bin/python3 -c from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=6, pipe_handle=8) --multiprocessing-fork


paper:

  - PyTorch Distributed: Experiences on Accelerating Data Parallel Training

    `<http://www.vldb.org/pvldb/vol13/p3005-li.pdf>`_
