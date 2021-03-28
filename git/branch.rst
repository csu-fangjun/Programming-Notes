
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


Remove unused remote branches locally
-------------------------------------

.. code-block::

   git remote prune origin

Create a new branch and switch to it
------------------------------------

.. code-block::

   git checkout -b <new_branch>

   # it is equvialent to the following two statements
   git branch <new_branch>
   git checkout <new_branch>

   # create a branch based on origin/master and check it out
   # it will also track the remote origin/master branch
   git checkout -b <new_branch> origin/master


Rename local branch
-------------------

.. code-block::

   git branch -m <old_name> <new_name>

   # and push it to the remote `origin`
   git push -u origin <new_name>
   git push origin --delete <old_name>

Useful commands
---------------

  git checkout -b create_some_new_branch
  git fetch some_origin
  git merge --ff some_origin/some_branch
