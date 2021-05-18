
Anaconda
========

Installation
------------

Go to `<https://www.anaconda.com/products/individual#linux>`_ and download the package:

.. code-block::

  wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
  chmod +x ./Anaconda3-2020.07-Linux-x86_64.sh
  ./Anaconda3-2020.07-Linux-x86_64.sh

  # default path is $HOME/anaconda3

  source ~/anaconda3/etc/profile.d/conda.sh
  ~/anaconda3/bin/conda create --name py38
  conda init bash
  conda activate py38
  conda install pytorch torchvision -c pytorch
  conda deactivate

  # to disable the automatic activation of base
  conda config --set auto_activate_base false

``~/.my-anaconda3.sh``:

.. code-block::

  #!/usr/bin/env bash

  eval "$(/root/fangjun/software/anaconda3/2020.11/bin/conda shell.bash hook)"
  # conda init
  # conda config --set auto_activate_base false


Usage::

  . ~/.my-anaconda3.sh
  conda info
  conda env -h
  # create an environment
  conda env create -h

  # create an environment with name k2
  conda env create -n k2 python=3.8

  # list available environment
  conda env list

  # activate an environment
  conda activate -h

  # activate the environment k2
  conda activate k2

  # deactivate the current environment
  conda deactivate

  # remove an environment
  conda env remove -h
  conda env remove -n k2

  conda env list
  conda activate <some_env_name>
  conda deactivate


Output of ``conda info``

.. code-block::

       active environment : None
              shell level : 0
         user config file : /root/fangjun/.condarc
   populated config files : /root/fangjun/.condarc
            conda version : 4.9.2
      conda-build version : 3.20.5
           python version : 3.8.5.final.0
         virtual packages : __cuda=11.1=0
                            __glibc=2.27=0
                            __unix=0=0
                            __archspec=1=x86_64
         base environment : /root/fangjun/software/anaconda3/2020.11  (writable)
             channel URLs : https://repo.anaconda.com/pkgs/main/linux-64
                            https://repo.anaconda.com/pkgs/main/noarch
                            https://repo.anaconda.com/pkgs/r/linux-64
                            https://repo.anaconda.com/pkgs/r/noarch
            package cache : /root/fangjun/software/anaconda3/2020.11/pkgs
                            /root/fangjun/.conda/pkgs
         envs directories : /root/fangjun/software/anaconda3/2020.11/envs
                            /root/fangjun/.conda/envs
                 platform : linux-64
               user-agent : conda/4.9.2 requests/2.24.0 CPython/3.8.5 Linux/5.4.54-1.0.0.std7c.el7.2.x86_64 ubuntu/18.04.5 glibc/2.27
                  UID:GID : 1088:0
               netrc file : None
             offline mode : False


The content of ``/root/fangjun/.condarc`` is::

  auto_activate_base: false

It contains only one line.

for k2
------

Gmail: ``k2fsa20@gmail.com``, passwd: ``k2-FSA/k2``, birth date: 1990.01.01

anaconda: user name ``k2-fsa``, email ``k2fsa20@gmail.com``., passwd: ``k2-FSA/k2``

github actions token: k2-bad08df7-1444-40d4-8467-ef274eb18285

Go to `<https://anaconda.org/k2-fsa/settings/access>`_ to create tokens.

Refer to `<https://docs.anaconda.com/anacondaorg/user-guide/tasks/work-with-accounts/>`
for more details.

.. code-block::

  anaconda upload /root/fangjun/software/anaconda3/2020.11/conda-bld/linux-64/click-7.0-py38_0.tar.bz2

  # It prints after authentication

  conda package located at:
  https://anaconda.org/k2-fsa/click

  # From the above page
  # conda install -c k2-fsa click


Refer to `<https://github.com/Anaconda-Platform/anaconda-client/issues/501#issuecomment-470742898>`_
in case upload fails without printing any messages.

