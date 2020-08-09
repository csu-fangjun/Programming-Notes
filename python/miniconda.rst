
Miniconda
=========

.. code-block::

  cd ~/software
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh

1. Press ``Enter``
2. Accept the license terms? yes
3. Install to location: $HOME/software/miniconda3
4. Do not run the initializer of Miniconda3.

The binary ``conda`` is in ``$HOME/software/miniconda3/bin/conda``. Add it to ``PATH``.

.. code-block:: console

  $ conda --version
  conda 4.8.2

To see the help of the some command use:

.. code-block::

  $ conda update --help
  $ conda env --help
  $ conda env create --help

Create a file `~/.activate-conda.sh`:

.. code-block::

    eval "$(/home/fangjunkuang/software/miniconda3/bin/conda shell.bash hook)"

If we want to use ``conda``, first run ``source ~/.activate-conda.sh``. It will
activate the ``(base)`` environment.

Run ``conda deactivate`` to revert it.


Create an Virtual Environment from a file
-----------------------------------------

Suppose the file is ``env.yml``:

.. code-block::

    name: mypy3
    channels:
      - conda-forge
      - defaults
    dependencies:
      - python==3.6
      - pip
      - pip:
          - pyjokes

Run

.. code-block:: console

  conda env create --file env.yml

The environment is installed to ``$HOME/software/miniconda3/envs/mypy3``.

To activate the environment, use:

.. code-block:: console

  source ~/.activate-conda.sh
  conda activate mypy3

To deactivate it, use:

.. code-block:: console

  conda deactivate

To view a list of installed environments, use:

.. code-block:: console

  $ conda env list

To remove an environment:

.. code-block::

  conda env remove --help
  conda env remove -n mypy3

It creates delete the directory ``$HOME/software/miniconda3/envs/mypy3``.

Create an Virtual Environment from commandline
-------------------------------------------

.. code-block::

  conda create -n myenv

It will install it to ``$HOME/software/miniconda3/envs/myenv`` using the default python version
that we are currently using.

Then use:

.. code-block::

  conda activate myenv
  conda deactivate
  conda env remove -n myenv


To use a specific python, run:

.. code-block::

  conda create -n myenv python==3.6.9

To install pip:

.. code-block::

  conda activate myenv
  conda install pip
  pip install pyyaml
