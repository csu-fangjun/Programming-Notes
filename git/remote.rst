
remote
======

.. code-block::

  git remote show origin

prints::

  * remote origin
    Fetch URL: git@github.com:csukuangfj/k2.git
    Push  URL: git@github.com:csukuangfj/k2.git
    HEAD branch: master
    Remote branches:
      fangjun-doc                                tracked
      fangjun-ragged-ops                         tracked
      master                                     tracked
      refs/remotes/origin/fangjun-benchmark      stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-fix            stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-fix-build      stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-fix-pythonpath stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-logger         stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-minor-fixes    stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-ragged-int     stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-row-splits     stale (use 'git remote prune' to remove)
      refs/remotes/origin/fangjun-sync-kernel    stale (use 'git remote prune' to remove)
    Local branches configured for 'git pull':
      fangjun-doc        merges with remote fangjun-doc
      fangjun-ragged-ops merges with remote fangjun-ragged-ops
      master             merges with remote master
    Local refs configured for 'git push':
      fangjun-doc        pushes to fangjun-doc        (local out of date)
      fangjun-ragged-ops pushes to fangjun-ragged-ops (up to date)
      master             pushes to master             (up to date)

After

.. code-block::

  git remote prune origin

prints::

  * remote origin
    Fetch URL: git@github.com:csukuangfj/k2.git
    Push  URL: git@github.com:csukuangfj/k2.git
    HEAD branch: master
    Remote branches:
      fangjun-doc        tracked
      fangjun-ragged-ops tracked
      master             tracked
    Local branches configured for 'git pull':
      fangjun-doc        merges with remote fangjun-doc
      fangjun-ragged-ops merges with remote fangjun-ragged-ops
      master             merges with remote master
    Local refs configured for 'git push':
      fangjun-doc        pushes to fangjun-doc        (local out of date)
      fangjun-ragged-ops pushes to fangjun-ragged-ops (up to date)
      master             pushes to master             (up to date)


Rename a branch
---------------

.. code-block::

  git remote rename <old_name> <new_name>

Remove a branch
---------------

.. code-block::

  git remote remove <name>
