
Submodules
==========


.. code-block::

  git submodule update --init --recursive


Add a submodule
---------------

.. code-block::

  git submodule add git@host:repo  ./external/repo

it will create two files:

- `.gitmodules`
- `external/repo`

.. code-block::

  cat .gitmodules
  [submodule "external/repo"]
  path = external/repo
  url = git@host:repo

Init a submodule
----------------

.. code-block::

    git submodule init
    git submodule update  # to get files from the remote

Remove a submodule
------------------

1. Manually modify `.gitmodules` to remove all files related to the submodule

2. Manually modify `.git/config` to remove all files related to the submodule if
we have run `git submodule init`

3. Remove the submodule folder: ``git rm --cached external/repo``

Update a submodule
------------------

.. code-block::

    git submodule init
    git submodule update
    cd external/repo
    git status
    git checkout master
    git pull
    cd ../..
    git status
    git add external/repo
    git status
    git commit

