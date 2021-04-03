SRILM
=====

Download
--------

  - `<https://github.com/wmeng223/srilm-package/raw/master/srilm-1.7.3.tar.gz>`_
  - `<http://www.speech.sri.com/projects/srilm/download.html>`_


Install
-------

.. code-block::

  wget https://github.com/wmeng223/srilm-package/raw/master/srilm-1.7.3.tar.gz
  mkdir srilm
  cd srilm
  tar xvf ../srilm-1.7.3.tar.gz
  # Read INSTALL inside the unziped directory
  #
  # Modify Makefile, set the variable `SRILM` to the absolute directory.
  #
  make -j

  # find . -name ngram-count

ngram-count
-----------

Refer to

  - `<https://pages.ucsd.edu/~rlevy/teaching/2015winter/lign165/lectures/lecture13/lecture13_ngrams_with_SRILM.pdf>`_

.. literalinclude:: ./code/srilm/corpus.txt
  :caption: code/srilm/corpus.txt
  :linenos:

.. code-block::

  ngram-count -text corpus.txt -order 2 -write corpus-bigram.count

.. literalinclude:: ./code/srilm/corpus-bigram.count
  :caption: code/srilm/corpus-bigram.count
  :linenos:

.. code-block::

  ngram-count -text corpus.txt -order 2 -addsmooth 0 -lm corpus-bigram-unsmooth.lm

.. literalinclude:: ./code/srilm/corpus-bigram-unsmooth.lm
  :caption: code/srilm/corpus-bigram-unsmooth.lm
  :linenos:

.. literalinclude:: ./code/srilm/bigram-unsmooth.py
  :caption: code/srilm/bigram-unsmooth.py
  :linenos:

The output of ``./code/srilm/bigram-unsmooth.py``:

.. literalinclude:: ./code/srilm/bigram-unsmooth-out.txt
  :caption: code/srilm/bigram-unsmooth-out.txt
  :linenos:
