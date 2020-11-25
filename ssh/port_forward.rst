
Port Forwarding
===============

.. code-block::

  # on some-machine
  ssh –R 2210:localhost:22 username@yourMachine.com

Open another terminal and logging into ``yourMachine.com``:

.. code-block::

  ssh -p 2210 user@localhost # it will connect to some-machine

In ``.ssh/config``, it looks like the following:

.. code-block::

  Host alias
    Hostname yourMachine.com
    User username
    IdentityFile ~/.ssh/xxx
    RemoteForward 2210 localhost:22


To bypass host verification:

.. code-block::

  ssh -o "StrictHostKeyChecking=no" user@host

To login with password

.. code-block::

  sudo apt-get install sshpass
  sshpass -p your_password ssh user@hostname

Install ssh server:

.. code-block::

  sudo apt-get install openssh-server
  sudo service ssh restart
  service ssh status
