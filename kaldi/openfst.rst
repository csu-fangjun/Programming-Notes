
OpenFST
=======

There is no GitHub for OpenFST. Go to its official website
for download: `<http://www.openfst.org/twiki/bin/view/FST/FstDownload>`_.

For installation, , read the file ``INSTALL`` contained in the downloaded file.

Example:

.. code-block:: bash

  wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.7.tar.gz
  tar xf openfst-1.7.7.tar.gz
  cd openfst
  ./configure --prefix=$HOME/software/openfst
  make -j10
  make install

It will create three directories: ``include``, ``lib`` and ``bin``.


Semiring
--------

It has two binary operations:

- :math:`\oplus`, for a Abelian group: associative, commutative, **identity**, inverse. The **identity** is denoted as ``Zero``.

- :math:`\otimes`, for a Monoid: associative, distributive w.r.t. `\oplus`. The **identity** is denoted as ``One``.


**left semiring**
  :math:`c\otimes (a\oplus b) = (c\otimes a) \oplus (c \otimes b)`

**right semiring**
  :math:`(a\oplus b) \otimes c = (a\otimes c) \oplus (b \otimes c)`

**semiring**
  It is a left semiring as well as a right semiring.

**commutative**
  :math:`a \otimes b = b \otimes a`

**idempotent**
  :math:`a \oplus a = a`.

  .. NOTE::

    Consider :math:`\oplus` as a ``min`` operation, so we have :math:`a \oplus a = a`

**left division**
  For :math:`a \otimes b = c`, left division is :math:`b = a^{-1} c`. In openfst, it is ``b = Divide(c, a, DIVIDE_LEFT)``.
  It is defined only for a semiring and a left semiring.

**right division**
  For :math:`a \otimes b = c`, right division is :math:`a = c b^{-1}`. In openfst, it is ``a = Divide(c, b, DIVIDE_RIGHT)``
  It is defined only for a semiring and a right semiring.

**natural order**
  :math:`a \leq b \iff a + b = a`

TropicalWeight
--------------

It is a subclass of ``template<typename T> class FloatWeightTpl;``.

``(min, +, inf, 0)`` is :math:`(\oplus, \otimes, Zero, One)`.

Note that:

- ``min`` == :math:`\oplus`
- ``+`` == :math:`\otimes`
- ``inf`` == ``Zero``
- ``0`` == ``One``

Algorithms
----------

- determinization
- minimization
- union
- intersection
- compaction
- composition


References
----------

- On some applications of finite-state automata theory to natural language processing :cite:`mohri1996some`

- Compact representations by finite-state transducers :cite:`mohri1994compact`

- Finite-state transducers in language and speech processing :cite:`mohri1997finite`

    By Mehryar Mohri.

    His google scholar profile is `<https://scholar.google.com/citations?user=ktwwLjsAAAAJ&hl=en&oi=sra>`_

    His own publication page: `<https://cs.nyu.edu/~mohri/pub/>`_

- Weighted automata in text and speech processing :cite:`mohri2005weighted`


- The design principles of a weighted finite-state transducer library :cite:`mohri2000design`

- OpenFst: A general and efficient weighted finite-state transducer library :cite:`allauzen2007openfst`

- https://github.com/opendcd/opendcd

- Introduction

  `<http://www.openfst.org/twiki/pub/FST/FstSltTutorial/part0.pdf>`_

  It lists lots of references.

- Part II. Library Use and Design

  `<http://www.openfst.org/twiki/pub/FST/FstHltTutorial/tutorial_part2.pdf>`_

  `<http://www.openfst.org/twiki/pub/FST/FstSltTutorial/part1.pdf>`_

- Part III. Applications

  `<http://www.openfst.org/twiki/pub/FST/FstHltTutorial/tutorial_part3.pdf>`_


  `<http://www.openfst.org/twiki/pub/FST/FstSltTutorial/part2.pdf>`_

- Generic :math:`\epsilon`-Removal and Input :math:`\epsilon`-Normalization Algorithms for Weighted
  Transducers

    https://cs.nyu.edu/~mohri/pub/ijfcs.pdf

- Weighted Transducer Algorithms

    `<https://cs.nyu.edu/~mohri/transducer-algorithms.html>`_


TODO
----

- https://www.cs.jhu.edu/~jason/465/hw-ofst/hw-ofst.pdf

