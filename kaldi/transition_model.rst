
Transition Model
================

Basic Terminology
-----------------


- phone: 1, 2, 3, ...

  Every phone has a list of Hmm State.

- Hmm State: 0, 1, 2, 3

  Every state, except the final non-emitting state, has a list of transitions.

  Every transition has a transition id and pdf id. Since pdf id can be shared,
  multiple transition ids can correspond to the same pdf id.

- transition state: 1, 2, 3, 4

  it is an index into a list of ``(phone, HmmState, forward_pdf_id, self_loop_pdf_id)``,

    transition state corresponds to ``(phone, HmmState)``; it corresponds to
    multiple pdf ids.

- transition id: 1, 2, 3, ...

  Every transition has an trantision id. Multiple transitions can correspond to the same pdf id.

  .. HINT::

    Number of transitions is larger than or equal to the number of pdf ids.









