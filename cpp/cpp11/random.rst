Random
======

.. caution::

  Use locks or thread local variables for thread safety!

.. caution::

  ``std::random_device`` is non-deterministic, since it reads
  the value from the device ``/dev/urandom``.

Things to take away:

  - How to get a seed?

    - Use a fixed number, e.g. 100

    - Construct an object ``std::random_device rd;`` and call ``rd()``,
      which returns a random number read from ``/dev/urandom``

    - Use ``std::seed_seq``. Its method ``generator`` produces
      an array of seeds.

  - How to get a random number?

    - Use ``std::mt19937 gen``. It can be initialized with a seed. The seed
      can be an integer or an object of ``std::seed_seq``.

      The call ``gen()`` returns a random number.

   - How to get a random number following some distribution

      - For instance, ``std::uniform_int_distribution`` + ``std::mt19937`` + ``std::random_device``.


UniformRandomBitGenerator
-------------------------

See `<https://en.cppreference.com/w/cpp/named_req/UniformRandomBitGenerator>`_.

It defines the requirements that a generator has to satisfy.

From the source code of libcxx ``include/random``, ``src/random.cpp``:

  - ``std::random_device(const std::string &__token = "/dev/urandom")``

    The argument is usually discarded in the constructor but it is checked
    that the value has to be ``/dev/urandom``.

  - The destructor does nothing.

  - ``uint32_t entropy() const`` always returns 0.

  - There are several implementations for ``uint32_t std::random_device::operator()() const``.

      1. Use ``getentropy``
      2. Use ``arc4random``
      3. Use ``nacl_secure_random``
      4. Read from ``/dev/urandom``


seed_seq
--------

There is a `std::vector<uint32_t> v` inside `seed_seq`.

The output of `seed_seq::generate` is *reproducible* and deterministic!

.. literalinclude:: ./code/seed_seq.cc
  :caption: seed_seq.cc
  :language: cpp
  :linenos:

mt11937
-------

11937 is a prime number, ``2^11937 - 1`` is also a prime number!
``mt11937`` is a typedef of a template and it has its own internal states
controlled by a seed. We have three ways to set the seed:

  - Set the seed in the constructor
  - Use its method ``seed``:

    - Provide the seed as an integer
    - Provide the seed via ``std::seed_seq``.

``mt19937`` is a generator returning integers! ``std::default_random_engine``
has similar interface to ``mt19937``.

.. literalinclude:: ./code/mt19937.cc
  :caption: mt19937.cc
  :language: cpp
  :linenos:

uniform_int_distribution
------------------------

There is no state in ``uniform_int_distribution``.
A distribution transform a given input according to a
particular probability function.

.. literalinclude:: ./code/uniform_int_distribution.cc
  :caption: uniform_int_distribution.cc
  :language: cpp
  :linenos:


References
----------

- Random Number Generation in C++11

    A paper by **Walter E. Brown**.

    `<https://isocpp.org/files/papers/n3551.pdf>`_

- `<https://channel9.msdn.com/Events/GoingNative/2013/rand-Considered-Harmful>`_

    A video by **Stephan T. Lavavej**

- `<https://github.com/effolkronium/random/blob/develop/include/effolkronium/random.hpp>`_

    Use of ``thread_local``.
