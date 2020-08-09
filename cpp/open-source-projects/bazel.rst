
Bazel
=====

.. code-block::

  bazel --output_base=/some/path //some/target:name


- `Configurable build attributes <https://docs.bazel.build/versions/master/configurable-attributes.html>`_

  To support customized keys for ``config_setting``, use ``define_values``. Note
  that ``values`` only supports one ``--define``, whereas ``define_values`` allows
  :qa
  :qa
  multiple ``--define``.

  +------------------+----------------------------------+---------------+
  | Common Options   | Values                           | Example Usage |
  +==================+==================================+===============+
  | cpu              | arm,x86,armeabi-v7a,k8           | --cpu=arm     |
  +------------------+----------------------------------+---------------+
  | compilation_mode | dbg,opt                          | -c dbg        |
  +------------------+----------------------------------+---------------+

  Note that ``k8`` cpu means ``x86_64`` [1]_.

  .. [1] https://en.wikipedia.org/wiki/AMD_K8


