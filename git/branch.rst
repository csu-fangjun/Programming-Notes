
Branch
======


Replace the current branch
--------------------------

.. code-block::

    git status  # it has to be in a clean status
    git reset --hard <target-branch>

After this, the current branch is the same with the ``target branch``.
This is useful to replace our local branch with the remote branch.


Delete remote branch
--------------------

.. code-block::

    git push origin --delete `<branch-name>`

