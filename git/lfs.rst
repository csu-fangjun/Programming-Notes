git lfs
=======

See `<https://www.atlassian.com/git/tutorials/git-lfs>`_
for more documentation.

See also `<https://sabicalija.github.io/git-lfs-intro/>`_.

Installation
------------

**Linux**:

.. code-block::

  sudo apt-get install git-lfs
  git lfs install

**macOS**:

.. code-block::

  brew install git-lfs
  git lfs install


git lfs install
---------------

``git lfs install`` adds the following content to ``~/.gitconfig``:

.. code-block::

  [filter "lfs"]
    clean = git-lfs clean -- %f
    smudge = git-lfs smudge -- %f
    process = git-lfs filter-process
    required = true


If we create a repo or clone a repo and run `git lfs install` inside
the repo, it creates the following files in ``.git/hooks``:

**pre-push**:

.. code-block:: bash

  #!/bin/sh
  command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting .git/hooks/pre-push.\n"; exit 2; }
  git lfs pre-push "$@"

**post-commit**:

.. code-block:: bash

  #!/bin/sh
  command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting .git/hooks/post-commit.\n"; exit 2; }
  git lfs post-commit "$@"

**post-merge**:

.. code-block:: bash

  #!/bin/sh
  command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting .git/hooks/post-merge.\n"; exit 2; }
  git lfs post-merge "$@"

**post-checkout**:

.. code-block:: bash

  #!/bin/sh
  command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nThis repository is configured for Git LFS but 'git-lfs' was not found on your path. If you no longer wish to use Git LFS, remove this hook by deleting .git/hooks/post-checkout.\n"; exit 2; }
  git lfs post-checkout "$@"


By default, run ``git lfs install`` and then run ``git clone``, it will download
LFS files from the remote server.

Use ``git lfs install --skip-smudge`` to disable that behavior.

We can also use ``git lfs install --local --skip-smudge``.


``git lfs install --local --skip-smudge`` makes the following changes:

.. code-block::

  .git/config       filter.lfs.smudge=git-lfs smudge --skip — %f
  .git/config       filter.lfs.process=git-lfs filter-process --skip
  .git/config       filter.lfs.required=true
  .git/config       filter.lfs.clean=git-lfs clean — %f


To pull single LFS tracked file, use:

.. code-block:: bash

  git lfs pull --include=filename

  # Both of them are OK
  git lfs pull --include="Linv.pt"
  git lfs pull --include="./data/lang_bpe_500/Linv.pt"

To pull all LFS files, use: ``git lfs pull``.

Before downloading ``L.pt``, its content is:

.. code-block::

  $ cat L.pt
  version https://git-lfs.github.com/spec/v1
  oid sha256:0b8c4d3be529e6436db51f0d65c3d5423a36a4bac58b5f2764f553255502f9e1
  size 19025793

After using ``git lfs pull --include=L.pt``:

.. code-block::

  $ git lfs pull --include=L.pt
  Git LFS: (1 of 1 files) 18.14 MB / 18.14 MB

.. code-block::

  $ git lfs pointer --file=L.pt
  Git LFS pointer for L.pt

  version https://git-lfs.github.com/spec/v1
  oid sha256:0b8c4d3be529e6436db51f0d65c3d5423a36a4bac58b5f2764f553255502f9e1
  size 19025793


git lfs track
-------------

To add a file to ``git lfs``, use:

.. code-block::

  git lfs track xxx.foo
  git add xxx.foo


Note: ``git lfs track "*.png"`` will create a file ``.gitattributes`` in the directory
where ``git lfs track`` is executed, so it is best to run ``git lfs track`` in
the root directory of the repo.

Note: It is ``"*.png"``, not, ``*.png``. Don't forget the double quotes.

To untrack a pattern, use ``git lfs untrack "*.png"``.

git lfs fetch
--------------

.. code-block::

  cd repo
  git remote add github git@github.com:xxx/xxx
  git remote add bitbucket git@bitbucket.org:xxx/xxx
  git lfs fetch --all github
  git push --mirror bucket
  git lfs push --all bucket

To download all git lfs histories, use:

.. code-block::

  # by default, recent means 7 days.
  git lfs fetch --recent

  # to change recent to mean 10 days
  git config lfs.fetchrecentrefsdays 10

  # To fetch all LFS files
  git lfs fetch --all


git lfs prune
-------------

.. code-block::

  # to delete local LFS cache
  git lfs prune

  git lfs prune --dry-run
  git lfs prune --dry-run --verbose
  git lfs prune --verify-remote


