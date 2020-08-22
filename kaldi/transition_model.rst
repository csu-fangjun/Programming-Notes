
TransitionModel
===============

EventMap
--------

The leaves are constant event map, which saves a single value;
its map function always returns the same value whatever the key is.

TableEvent
----------

`std::map<int, EventMap*>`, key is an int and the value is an
``EventMap`` pointer. When the key type is ``-1``, the ``int`` of the
map designates pdf classes can can be 0, 1, 2, ...; when the key type
is ``0``, or ``1``, ..., which means the position of the phone,
the ``int`` of the map represents the phone id in the phones.txt.
Note that phone id is greater than 0. The eps is not in the tree.

SplitEvent
----------

Its key type is usually greater than -1. It has a yes set,
a yes map, and a no map.

HmmTopology
-----------

- ``HmmState``, forward pdf class, self loop pdf class, a vector of transitions; every transition has  a dest state and a probability.

- ``TopologyEntry``, a vector of ``HmmState``.


TransitionModel
---------------

**shared**
  - state 0 of phone 1 and phone 2 share the same pdf id.
  - state 1 of phone 1 and phone 2 share the same pdf id.
  - state 2 of phone 1 and phone 2 share the same pdf id.
  - state 3 of phone 1 and phone 2 share the same pdf id.

  Or

  - state 0 and state 1 of phone 1 share the same pdf id.

Given a pdf id, it may map to:

- a list of (phone, pdf class)
- a list of (phone, hmm state)

All information a transition model has is in a vector of ``tuple``:
(phone, hmm state, forward pdf id, self loop pdf id). Note that
it is pdf id, not pdf classes!

Suppose the tuple is a vector of 3 elements:

.. code-block::

  (1, 0, 10, 10)    // tstate == 1
  (1, 1, 11, 11)    // tstate == 2
  (1, 2, 12, 12)    // tstate == 3
  (1, 3, 13, 13)    // tstate == 4

``tstate`` means transition state, which is an index to the vector of tuples.
Note that it starts from 1.

(1, 0, 10, 10) means the phone is 1, hmm state is 0, forward pdf id is 10, and
self loop pdf id is 10. Suppose each hmm state has 3 transitions. Define an array:

.. code-block::

  std::vector<int> state2id_(1 + 4 + 1);


``state2id`` means transition state to transition id. Note that transition id starts from 1.

``1+4+1``: the first number is ``1`` since we starts from 1; the second number is 4 since
the tuple vector has 4 elements; the last number is 1 since ``state2id_[i+1] - state2id_[i]``
indicates the number of transitions for tstate ``i``.

- ``state2id[1]`` is 1; because hmm state 0 has 3 transitions
- ``state2id[2]`` is 4, which is ``state2id[1] + 3``
- ``state2id[3]`` is 7, which is ``state2id[2] + 3``
- ``state2id[4]`` is 10, which is ``state2id[3] + 3``
- ``state2d[5]`` is 13.

- Number of transitions for tstate 1: ``state2id[2] -  state2id[1] = 3``
- Number of transitions for tstate 2: ``state2id[3] -  state2id[2] = 3``
- Number of transitions for tstate 3: ``state2id[4] -  state2id[3] = 3``
- Number of transitions for tstate 4: ``state2id[5] -  state2id[4] = 3``

.. code-block::

  std::vector<int> id2state(13); // transition id starts from 1, there are 12 transitions

``id2state`` means transition id to transition state.

- ``id2state[0] = 1``
- ``id2state[1] = 1``
- ``id2state[2] = 1``
- ``id2state[3] = 2``
- ``id2state[4] = 2``
- ``id2state[5] = 2``
- ``id2state[6] = 3``
- ``id2state[7] = 3``
- ``id2state[8] = 3``
- ``id2state[9] = 4``
- ``id2state[10] = 4``
- ``id2state[11] = 4``

.. code-block::

  std::vector<int> id2pdf_id(12); // transition id starts from 1, there are 12 transitions


``id2pdf_id`` means transition id to pdf id.

- ``id2pdf_id[1] = 10``
- ``id2pdf_id[2] = 10``
- ``id2pdf_id[3] = 10``
- ``id2pdf_id[4] = 11``
- ``id2pdf_id[5] = 11``
- ``id2pdf_id[6] = 11``
- ``id2pdf_id[7] = 12``
- ``id2pdf_id[8] = 12``
- ``id2pdf_id[9] = 12``
- ``id2pdf_id[10] = 13``
- ``id2pdf_id[11] = 13``
- ``id2pdf_id[12] = 13``

To view the transition model in text format, use

.. code-block::

  copy-transition-model --binary=false final.mdl transition_model.txt

Part of the output is shown below:

.. code-block::

    <Triples> 653
    1 0 0
    1 1 1
    1 2 2
    1 3 3
    1 4 4
    2 0 5
    2 1 6
    2 2 7
    3 0 5
    3 1 6
    3 2 7
    4 0 5
    4 1 6
    4 2 7


Another example output:

.. code-block::

    <Tuples> 3227
    1 0 0 217
    2 0 1 2805
    2 0 543 2634
    2 0 1090 1620
    2 0 1151 850
    2 0 1386 1263
    2 0 1819 3396
    2 0 3279 3703
    2 0 3578 274
    2 0 3578 2634
    2 0 3578 2805
    2 0 3578 3703
    2 0 4133 2634
    2 0 4133 3703

Note that transition ids start from 1 because they are used
as input labels in FST. 0 is for epsilon in FST. Pdf ids start from 0.
