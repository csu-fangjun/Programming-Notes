nvtx
====

Installation (macOS)
--------------------

Visit `<https://developer.nvidia.com/gameworksdownload#?dn=cuda-toolkit-developer-tools-for-macos-11-3-0>`_.

Download `NsightSystems-macos-public-2021.2.1.58-642947b.dmg`. After downloading,
click it and drag it to the folder ``Applications``.


See `<https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx>`_ for how to use NVTX in code.
Refer to `k2/k2/csrc/nvtx.h` for a minimal usage case of NVTX.

After generating a file ``report1.qdrep``, we can copy it to macOS and open it with NVIDIA NsightSystems.

