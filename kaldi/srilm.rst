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

PPL
---

First example, which contains only one line:

.. literalinclude:: ./code/srilm/test1.txt
  :caption: code/srilm/test1.txt
  :linenos:

.. code-block::

  p(<s> hello world </s>) = p(hello|<s>) p(world|hello) p(</s>|world)
                          = 1/5 * 2/3 * 2/4
                          = 1/15
  math.log10(1/15) = - 1.176091
  ppl = (15)^(1/3) = math.pow(15, 1/3.) = 2.46621207433047

.. code-block::

  ngram -lm corpus-bigram-unsmooth.lm -ppl test1.txt

The output is:

.. code-block::

  $ ngram -lm corpus-bigram-unsmooth.lm -ppl test1.txt
  file test1.txt: 1 sentences, 2 words, 0 OOVs
  0 zeroprobs, logprob= -1.176091 ppl= 2.466212 ppl1= 3.872984

Second example, which contains only one line:

.. literalinclude:: ./code/srilm/test2.txt
  :caption: code/srilm/test2.txt
  :linenos:

.. code-block::

  p(<s> hello world bar foo </s>) = p(hello|<s>) p(world|hello) p(bar|world) p(foo|bar) p(</s>|foo)
                          = 1/5 * 2/3 * 2/4 * 1/3 * 1/4
                          = 1/180
  math.log10(1/180) = - 2.255272505103306
  ppl = (180)^(1/5) = math.pow(180, 1/5.) = 2.825234500494767

.. code-block::

  ngram -lm corpus-bigram-unsmooth.lm -ppl test2.txt

The output is:

.. code-block::

  $ ngram -lm corpus-bigram-unsmooth.lm -ppl test2.txt
  file test2.txt: 1 sentences, 4 words, 0 OOVs
  0 zeroprobs, logprob= -2.255273 ppl= 2.825235 ppl1= 3.662842

Third example, which contains two lines:

.. literalinclude:: ./code/srilm/test3.txt
  :caption: code/srilm/test3.txt
  :linenos:

.. code-block::


  p(<s> hello world </s>) = p(hello|<s>) p(world|hello) p(</s>|world)
                          = 1/5 * 2/3 * 2/4
                          = 1/15

  p(<s> hello world bar foo </s>) = p(hello|<s>) p(world|hello) p(bar|world) p(foo|bar) p(</s>|foo)
                          = 1/5 * 2/3 * 2/4 * 1/3 * 1/4
                          = 1/180

  p(two sentences) = 1/15 * 1/180 = 1/2700
  math.log10(1/2700) = -3.4313637641589874
  math.pow(2700, 1/(3+5)) = 2.6848527412884793

  The first sentence contains 3 words and the second contains 5 words.
  Note that </s> is counted as one extra word for each sentence.

.. code-block::

  ngram -lm corpus-bigram-unsmooth.lm -ppl test3.txt

The output is:

.. code-block::

  $ ngram -lm corpus-bigram-unsmooth.lm -ppl test3.txt
  file test3.txt: 2 sentences, 6 words, 0 OOVs
  0 zeroprobs, logprob= -3.431364 ppl= 2.684853 ppl1= 3.731591
