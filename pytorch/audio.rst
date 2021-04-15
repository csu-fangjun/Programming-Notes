
Audio
=====

.. code-block::

  git clone https://github.com/pytorch/audio.git


sox
---

The manual `<https://www.systutorials.com/docs/linux/man/1-sox/>`_.

An effects chain is terminated by placing a : (colon) after an effect.
Any following effects are a part of a new effects chain.

generate
^^^^^^^^

.. code-block::

  sox -n -r 16k -b 16 a.wav synth 1 sine 100

- ``-n`` mean null file; it has no input.
- ``-r`` means ``--rate``, sample rate in Hz
- ``-b`` means ``--bits``, number of bits for each sample
- ``synth`` means synthesize

1 means 1 second; ``sine 100`` means 100 Hz sine wave.


play
^^^^

.. code-block::

  play input.wav
  # or
  sox input.wav -d

record
^^^^^^

.. code-block::

  rec new_file.wav
  # or
  sox -d new_file.wav

convert format
^^^^^^^^^^^^^^

.. code-block::

  # conversion from .au to .wav
  sox input.au output.wav

  sox -r 16k -e signed -b 8 -c 1 voice-memo.raw voice-memo.wav


speed
^^^^^

.. code-block::

  sox input.wav output.wav speed 1.027

concatenate
^^^^^^^^^^^

.. code-block::

  sox left_part_in.wav right_parth.wav output_whole_part.wav

The length of the output is the sum of the two inputs.
If input1.wav is 1 second, input2.wav is 2 seconds, then
output.wav is 3 seconds.

mix
---

.. code-block::

  sox -m input1.wav input2.wav output.wav

If input1.wav is 1 second, input2.wav is 2 seconds, then
output.wav is 2 seconds.
