arpa
====

kaldi's arpa2fst
----------------

a line in arpa file looks like::

  log10(p)  word2 word3  log10(backoff)
  log10(p)  word1 word2 word3  log10(backoff)

.. Caution::

  arpa files uses ``log10(p)``, but FSA uses ``-ln(p)``.

  ``ln(p) = ln(10) * log10(p)``

To compute the probability of P(w1 w2 w3):

  - if P(w1, w2, w3) exists, then return it
  - r = back off probability of P(w1, w2)
  - return r + P(w2, w3)
