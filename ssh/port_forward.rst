
Port Forwarding
===============

.. code-block::

  # on some-machine
  ssh -R 2210:localhost:22 username@yourMachine.com
  # it listens on port 2210 on yourMachine and forward it to some-machine:22

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

ssh-copy-id::

  ssh-copy-id -i some_key some_server
  # equivalent to go to some_server/.ssh
  # cat local_host/.ssh/some_key.pub  some_server/.ssh/authorized_keys


Login remote server via a proxy::

  Host my_proxy
    Hostname ip_for_my_proxy
    User my_name_at_my_proxy
    IdentityFile ~/.ssh/my_key_at_my_proxy

  # log into remote_server via a proxy
  # if we cannot access remote server locally
  Host remote_server
    ProxyCommand ssh my_proxy nc %h %p
    User my_name_at_remote_server
    IdentityFile ~/.ssh/my_key_at_remote_server

