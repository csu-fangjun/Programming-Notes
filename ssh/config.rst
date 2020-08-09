

Passwordless Login
==================

Generate keys
-------------

Use ``ssh-keygen -t rsa`` to generate a pair of keys.

.. code-block:: console

    $ ssh-keygen -t rsa

    Generating public/private rsa key pair.
    Enter file in which to save the key (/path/to/home/.ssh/id_rsa): ./foo
    Enter passphrase (empty for no passphrase):
    Enter same passphrase again:
    Your identification has been saved in ./foo.
    Your public key has been saved in ./foo.pub.
    The key fingerprint is:
    SHA256:qGvftxOod4mwI1nIwrclRHuX1fBb6DVMZ9BQPtLQewo user@server
    The key's randomart image is:
    +---[RSA 2048]----+
    |          .o .==+|
    |    .     ...+o=.|
    |   . .   o  o.=oo|
    |    o ..o  .E+.oo|
    | . o o..S.  o. ..|
    |  o =.+ . .   .  |
    |   o.* + . o     |
    |    =.+.o =      |
    |   ..o.o.o.o     |
    +----[SHA256]-----+


Copy Keys
---------

Copy the generated public key to the remote server:

.. code-block:: bash

    ssh-copy-id -i /path/to/foo user@server


.. Note::

  Even if we use ``/path/to/foo``, it is the public key ``foo.pub``
  that is copied to the server.

Edit .ssh/config
----------------

.. code-block:: bash
  :caption: .ssh/config

  Host alias
    Hostname <server-ip>
    IdentityFile /path/to/foo
    User <username>
    Port <port>


Configuration for Git
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash
  :caption: .ssh/config

  Host github.com
    Hostname github.com
    User git
    IdentityFile /path/to/foo

  Host github2
    Hostname github.com
    User git
    IdentityFile /path/to/foo

.. HINT::

  If we have different GitHub accounts or an account
  associated with different organizations, we can setup
  different keys for different accounts or organizations.

.. code-block:: bash
  :caption: .git/config

  [remote "origin"]
    url = git@github.com:user/repo.git
    fetch = +refs/heads/*:refs/remotes/origin/*

  [remote "foobar"]
    url = git@github2:user2/repo.git
    fetch = +refs/heads/*:refs/remotes/foobar/*


Useful Options
--------------

- Disable host key checking::

    StrictHostKeyChecking=no

- Force password only login::

    PreferredAuthentications=password
    PubkeyAuthentication=no

- Force public key login only on the server side, changing ``/etc/ssh/sshd_config``::

    PasswordAuthentication no
    UsePAM no


References
----------

- Configuring ssh to access lab machines `<http://cit.dixie.edu/cs/2810/ssh-config.php>`_

    It shows how to configure a ssh tunnel to by-pass the firewall.

