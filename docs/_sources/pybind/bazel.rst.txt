
Build with Bazel
================

As of today (2020-03-15), the latest bazel version
is ``2.2.0``. Use the following commands to install it:

.. code-block:: bash

    wget https://github.com/bazelbuild/bazel/releases/download/2.2.0/bazel-2.2.0-installer-linux-x86_64.sh
    mkdir ~/software/bazel/2.2.0
    chmod +x bazel-2.2.0-installer-linux-x86_64.sh
    ./bazel-2.2.0-installer-linux-x86_64.sh --prefix=$HOME/software/bazel/2.2.0
    export PATH=$HOME/software/bazel/2.2.0/bin
    echo "source $HOME/software/bazel/2.2.0/lib/bazel/bin/bazel-complete.bash" >> $HOME/.bashrc

After installation, ``bazel version`` should print the following::

    Build label: 2.2.0
    Build target: bazel-out/k8-opt/bin/src/main/java/com/google/devtools/build/lib/bazel/BazelServer_deploy.jar
    Build time: Tue Mar 3 09:26:12 2020 (1583227572)
    Build timestamp: 1583227572
    Build timestamp as int: 1583227572

To compile the following code,

.. literalinclude:: ./code/hello.cc
  :caption: hello.cc
  :language: cpp
  :linenos:

.. literalinclude:: ./code/WORKSPACE
  :caption: WORKSPACE
  :language: python
  :linenos:

.. literalinclude:: ./code/.bazelrc
  :caption: .bazelrc
  :language: python
  :linenos:

.. literalinclude:: ./code/BUILD.tpl
  :caption: BUILD.tpl
  :language: python
  :linenos:

.. literalinclude:: ./code/python_repo.bzl
  :caption: python_repo.bzl
  :language: python
  :linenos:

.. literalinclude:: ./code/BUILD.bazel
  :caption: BUILD.bazel
  :language: python
  :linenos:

.. literalinclude:: ./code/test.py
  :caption: test.py
  :language: python
  :linenos:
