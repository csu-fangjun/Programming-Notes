
Rebase
======

Change commit message
---------------------

**Change last commit**
  .. code-block::

      git commit --amend

**Change last N commit**
  .. code-block::

      git rebase -i HEAD~n

  It shows::

    pick xxx  First commit
    pick xxx  ............
    pick xxx  Latest commit

  Replace ``pick`` with ``reword`` (follow the comment in the editor) and press ``:wq``.
  A separate window is poped out for editing messages.

