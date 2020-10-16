
Lattices
========


Compact Lattices
----------------

- LatticeWeight: value1 (graph cost), value2 (acoustic cost), defined in
  ``fstext/lattice-weight.h`` and ``lat/kaldi-lattice.h``

- CompactLatticeWeight: it has two parts: LatticeWeight and ``std::vector<int32_t>``.
  Its ``String`` method returns ``std::vector<int32_t>``, which is a vector of
  transition IDs. This represents alignment. It can be converted to phones
  using function ``SplitToPhones`` from ``hmm/hmm-utils.h``.

- CompactLatticeArc: it is an acceptor, ilable == olabel == word_id

This blog `http://codingandlearning.blogspot.com/2014/01/kaldi-lattices.html`_
has some text forms of CompactLattices.
