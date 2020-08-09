
Sox
===

Generate a wav file:

.. code-block:: bash

  sox -n -r 16000 -b 16 /tmp/test.wav synth 1 sine 100

- ``-n`` mean null file; it has no input.
- ``-r`` means ``--rate``, sample rate in Hz
- ``-b`` means ``--bits``, number of bits for each sample
- ``synth`` means synthesise

  1 means 1 second; ``sine 100`` means 100 Hz sine wave.

``soxi /tmp/test.wav`` prints::

    Input File     : '/tmp/test.wav'
    Channels       : 1
    Sample Rate    : 16000
    Precision      : 16-bit
    Duration       : 00:00:01.00 = 16000 samples ~ 75 CDDA sectors
    File Size      : 32.0k
    Bit Rate       : 256k
    Sample Encoding: 16-bit Signed Integer PCM

References
----------

- Sox in phonetic research `<http://linguistics.berkeley.edu/plab/guestwiki/index.php?title=Sox_in_phonetic_research>`_

