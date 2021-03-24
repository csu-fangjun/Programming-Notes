
Lhotse notes
============

AudioSource
-----------

It represents a WAVE file and supports only one method ``load_audio``.
Its ``channels`` and ``type`` are from the config yaml file.

.. NOTE::

  It does not store the wave data. The data is read from a file or
  some command output.

Recording
---------

A recording is a list of ``AudioSource`` with extra meta information, such
as ``sampling_rate``, ``num_samples``, ``duration_seconds``, and ``id``.
It is the user's responsibility that on audio source has the same channel id.

AudioSet
--------

It contains a list of ``Recording`` and support reading/writing yaml files.

