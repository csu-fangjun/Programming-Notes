macOS
=====

system services
---------------

See `<https://gist.github.com/marcelaraujo/9a9fe07c5a4bcaea8c06>`_.

- system

    .. code-block::

      sudo ls -all /System/Library/LaunchDaemons/

- third party

    .. code-block::

      sudo ls -all /Library/LaunchDaemons/

To disable some file:

Run::

  sudo launchctl unload -w /System/Library/LaunchDaemons/file.plist


If it complains `register-python-argcomplete` not found, run:

.. code-block::

  pip3 install --user bash-completion

