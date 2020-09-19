
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
