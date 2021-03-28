Install
=======

Refer to `<https://golang.org/doc/install>`_.

.. code-block::

  wget https://golang.org/dl/go1.16.2.linux-amd64.tar.gz
  cd ~/software
  tar xvf go1.16.2.linux-amd64.tar.gz
  mv go go-1.16.2
  ln -s $PWD/go-1.16.2 go
  export PATH=$HOME/software/go/bin:$PATH

.. code-block::

  go version

print::

  go version go1.16.2 linux/amd64
