
Config
======

.. code-block::

    git config --list           # .git/config
    git config --list --global  # ~/.gitconfig
    git config --list --system  # /etc/gitconfig

    git config user.name "Your Name"
    git config user.email "your@email.com"

    # if using https, the following settings will cache the password
    git config --global credential.helper cache
