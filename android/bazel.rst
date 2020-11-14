
Bazel
=====

* Installation via bazelisk::

   wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.4/bazelisk-linux-amd64
   mv bazelisk-linux-amd64 $HOME/bin/software/bazel
   export PATH=$HOME/software/bin:$PATH

  The drawback of bazelisk is that it does not include the bash completion script.

* Instatllion via binary::

   wget https://github.com/bazelbuild/bazel/releases/download/3.7.0/bazel-3.7.0-installer-linux-x86_64.sh
   chmod +x bazel-3.7.0-installer-linux-x86_64.sh
   ./bazel-3.7.0-installer-linux-x86_64.sh --help
   ./bazel-3.7.0-installer-linux-x86_64.sh --prefix=$HOME/software/bazel-3.7.0
   export PATH=$HOME/software/bazel-3.7.0/bin:$PATH

   # for bash completion
   source $HOME/software/bazel-3.7.0/lib/bazel/bin/bazel-complete.bash
