
Colab
=====

`<https://pypi.org/project/colab-ssh/>`_.

Go to `<https://dashboard.ngrok.com/auth>`_ to register
an account. Get the token:

.. code-block::

  1eIJYedmsAUrNHWOTCntHFBcn8P_2LKRAzhBevRkiGK5PUNqQ


.. code-block::

  !pip install colab_ssh --upgrade

  from colab_ssh import launch_ssh
  launch_ssh('1eIJYedmsAUrNHWOTCntHFBcn8P_2LKRAzhBevRkiGK5PUNqQ','optional-password')

