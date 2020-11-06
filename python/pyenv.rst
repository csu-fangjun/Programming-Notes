
pyenv
=====

.. code-block::

  if [ ! -d $HOME/open-source/pyenv ]; then
    cd $HOME/open-source
    git clone https://github.com/pyenv/pyenv.git
    echo 'export PYENV_ROOT="$HOME/open-source/pyenv"' >> ~/.fangjun.sh
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.fangjun.sh
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.fangjun.sh
  else
    echo "Done"
  fi

We have to install the following dependencies according to
`<https://github.com/pyenv/pyenv/wiki>`_

.. code-block::

  sudo apt-get update \
  sudo apt-get install --no-install-recommends make build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget \
    curl llvm libncurses5-dev xz-utils libxml2-dev libxmlsec1-dev \
    libffi-dev liblzma-dev

To list available python versions for installing, use ``pyenv install -l``.

To install python 3.7, use:

.. code-block::

  pyenv install 3.7.0

To install with shared library, use

.. code-block::

  PYTHON_CONFIGURE_OPTS=--enable-shared pyenv install 3.8


To view installed versions, use:

.. code-block::

  pyenv versions

To view the current version in use, use:

.. code-block::

  pyenv global

To switch to python 3.7.0, use:

.. code-block::

  pyenv global 3.7.0

The installed python environment is in ``open-source/pyenv/versions/3.7.0``.
