
Bazel
=====

- `Configurable build attributes <https://docs.bazel.build/versions/master/configurable-attributes.html>`_

  To support customized keys for ``config_setting``, use ``define_values``. Note
  that ``values`` only supports one ``--define``, whereas ``define_values`` allows
  :qa
  :qa
  multiple ``--define``.

  +------------------+------------+---------------+
  | Common Options   | Values     | Example Usage |
  +==================+============+===============+
  | cpu              | arm,x86    | --cpu=arm     |
  +------------------+------------+---------------+
  | compilation_mode | dbg,opt    | -c dbg        |
  +------------------+------------+---------------+
