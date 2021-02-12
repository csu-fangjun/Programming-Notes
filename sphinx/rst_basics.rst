
reStructuredText Basics
=======================

Heading style:

  - parts: ``#``
  - chapters: ``*``
  - sections: ``=``
  - subsections: ``-``
  - subsubsections: ``^``
  - paragraphs: ``"``


URLs:

- `Link to Google <https://google.com>`_
- `Another link to Google`_

.. _Another link to Google: https://google.com


.. code-block::

    .. _recommend-saving-models:

    Recommended approach for saving a model
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


See `<https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html>`_
for adding links.

Cross references in code
------------------------

.. code-block::

  :class:`Variable`
  :class:`tuple`
  :meth:`~Function.forward`
  :attr:`~Variable.needs_input_grad`
  :any:`python:None`
  :mod:`torch.nn`
  :mod:`~torch.nn`
  :func:`~Function.backward`
  :class:`torch.Tensor`


References
----------

- Read the Docs Sphinx Theme [1]_
- `reStructuredText Primer`_




Footnotes
---------

.. [1] `Read the Docs Sphinx Theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/index.html>`_
.. _reStructuredText Primer: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
