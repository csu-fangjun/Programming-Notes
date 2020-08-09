
Fbank
=====

Before computing fbank, we have to know the following parameters:

**sample rate**
  It is usually 16 kHz, which means there are 16 samples per 1 ms.

**frame shift**
  For a 16 kHz wav, it is usually 10 ms. That is,  160 samples.

**frame length**
  For a 16 kHz wav, it is usually 25 ms, i.e., 400 samples. To compute FFT,
  we have to pad 112 zeros to get 512 samples.

**num bins**
  It is usually 23.


If a wav file has N samples, how many number of frame are there for fbank?

  number of frames = :math:`\left\lfloor\frac{N - 400}{160}\right\rfloor + 1` [1]_



Steps to compute fbank
----------------------

1. Extract one frame ``x`` from a wav file, where ``len(x) == 400``.

2. If dither is enabled, then apply dither [2]_:

    x[i] = x[i] + rand() * dither_value

    where `rand()` returns a guassian distributed variable with mean 0 and standard deviation 1.
    ``dither_value`` is default to 1 in Kaldi.


3. Remove DC such that ``mean(x) == 0`` [3]_

    x[i] = x[i] - mean(x)

4. Pre-emphasis [4]_

    x[i] = x[i] - preemph_coeff * x[i - 1]  for i = 399, 398, ..., 1

    x[0] = x[0] - preemp_coeff * x[0]

    where ``preemph_coeff`` is usually 0.97

5. Apply window.

    x[i] = x[i] * w[i]

    There are several window functions in Kaldi, which are summarized below [5]_:

    **Hanning**
      .. math::

          w[i] = 0.5 - 0.5  \cos \big(2  \pi  \frac{i}{N-1}\big)

      where ``N`` is 400 in our case and ``i`` is from 0 to ``N-1``.

    **Hamming**
      .. math::

          w[i] = 0.54 - 0.46  \cos \big(2  \pi  \frac{i}{N-1}\big)

    **Povey**
      .. math::

          w[i] = \Big(0.5 - 0.5  \cos \big(2  \pi  \frac{i}{N-1}\big)\Big)^{0.85} = \mathrm{Hanning}^{0.85}

      Note that ``Povey`` is named after Daniel Povey [6]_ and this is the default window type in Kaldi.

    **Blackman**:
      .. math::

        w[i] = 0.42 - 0.5 * \cos\big(2  \pi \frac{i}{N - 1}\big) + 0.08  \cos\big(4 \pi \frac{i}{N - 1}\big)

6. Apply padding [7]_.

    Pad the frame with zeros so that there are 512 samples in the frame.

7. Apply FFT [8]_.

8. Compute power spectrum [9]_.

    - real0 * real0
    - real1 * real1 + img1 * imag1
    - real2 * real2 + img2 * img2
    - ... ...
    - real255 * real255 + img255 * img255
    - Note that real256 is **NOT** used.


    Now we have 256 non-negative real numbers.

9. Apply Melbank.

   After applying Melbank, we get ``num_bin`` real numbers, e.g., 23.

   Its essence is a GEMV. We describe in the following how to obtain rows of the matrix.

   To convert Hz to mel, we can use the following equation [10]_:

   .. math::

    mel = 1127 \log \left( 1 + \frac{hz}{700} \right)

   There are several parameters in Mel bank computation:

   **low frequency**
    Default value is 20 Hz.

   **high frequency**
    Default value is 0.5 * 16 kHz = 8 kHz

   **fft bin width**
     :math:`\frac{16 kHz}{512}` = 31.25 Hz

   **mel low frequence**
    Convert `low frequence` from Hz to mel.

   **mel high frequence**
    Convert `high frequence` from Hz to mel.

   **mel frequence delta**
    .. math::

      \frac{\mathrm{mel\ high\ frequency} - \mathrm{mel\ low\ frequency}}{23 + 1}

    where 23 is the number of mel bins.

   Each row of the matrix contains the coefficient of a mel bin, which is very sparse.
   If ``num_bins==23``, then the matrix has 23 rows.

   .. code-block::

      for (int bin = 0; bin < num_bins; ++bin) {
        float left_mel = mel_low_freq + bin * mel_feq_delta;
        float center_mel = mel_low_freq + (bin + 1) * mel_feq_delta;
        float right_mel = mel_low_freq + (bin + 2) * mel_feq_delta;

        std::vector<float> coeff_for_this_bin(num_bins, 0);
        int first_index = -1;
        int last_index = -1
        // num_fft_bins is 512/2 = 256
        for (int i = 0; i < num_fft_bins; ++i) {
          float freq_hz = fft_bin_width * i;
          float mel = ConvertHzToMel(freq_hz);
          if ((left_mel < mel) && (mel < right_mel)) {
            float coeff;
            if (mel < center_mel) {
              coeff = (mel - left_mel) / (center_mel - left_mel);
            } else {
              coeff = (right_mel - mel) / (right_mel - center_mel);
            }

            coeff_for_this_bin[i] = weight;
            if (first_index == -1) {
              first_index = i;
            }
            last_index = i;
          }
        }

        assert(first_index != -1 && last_index >= first_index);

        // now we have scanned all the fft bins and the weights are contained in
        // `coeff_for_this_bin`. It is non-zero only in the interval [first_index, last_index]
        //
        // We can assign `coeff_for_this_bin` to one row of the matrix.
        //
        // In Kaldi, it only saves the non-zero weight. So it needs to save `first_index`.
      }

10. Apply log.

      For the above ``23`` numbers, take log for each of them.

      .. HINT::

        Before taking log, use ``x[i] = std::max(x[i], std::numeric_limits<float>::epsilon());``


.. NOTE::

  fbank and log-fbank are used interchangably.

Notes about FFT
---------------

**Fourier Transform**

  According to Wikipedia `<https://en.wikipedia.org/wiki/Discrete_Fourier_transform>`_, the formula
  for Fourier transform is

  .. math::

      X_k = \sum_{n=0}^{N-1} x_n e^{-\frac{i\cdot 2\pi}{N}k\cdot n}

  which can be written in a GEMV, where ``M`` is

  .. math::

      \mathbf{M} =
      \begin{pmatrix}
        e^{\frac{i\cdot 2\pi}{N}0 \cdot 0} &
        e^{\frac{i\cdot 2\pi}{N}0 \cdot 1} &
        e^{\frac{i\cdot 2\pi}{N}0 \cdot 2} &
        \ldots &
        e^{\frac{i\cdot 2\pi}{N}0 \cdot (N-1)} \\
        %
        e^{\frac{i\cdot 2\pi}{N}1 \cdot 0} &
        e^{\frac{i\cdot 2\pi}{N}1 \cdot 1} &
        e^{\frac{i\cdot 2\pi}{N}1 \cdot 2} &
        \ldots &
        e^{\frac{i\cdot 2\pi}{N}1 \cdot (N-1)} \\
        %
        \ldots & \ldots & \ldots & \ldots & \ldots \\
        %
        e^{\frac{i\cdot 2\pi}{N}(N-1) \cdot 0} &
        e^{\frac{i\cdot 2\pi}{N}(N-1) \cdot 1} &
        e^{\frac{i\cdot 2\pi}{N}(N-1) \cdot 2} &
        \ldots &
        e^{\frac{i\cdot 2\pi}{N}(N-1) \cdot (N-1)} \\
      \end{pmatrix}


The following code demonstrates FFT computation using GMEV and ``np.fft.fft``.

.. literalinclude:: ./code/test_fft.py
  :caption: test_fft.py
  :language: python
  :linenos:

The documentation for ``np.fft.fft`` can be found at
`<https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fft.html#numpy-fft-fft>`_.



.. [1] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-window.cc#L54>`_
.. [2] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-window.cc#L97>`_
.. [3] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-window.cc#L147>`_
.. [4] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-window.cc#L105>`_
.. [5] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-window.cc#L116>`_
.. [6] `<https://github.com/danpovey>`_
.. [7] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-window.cc#L188>`_
.. [8] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-fbank.cc#L88>`_
.. [9] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/feature-functions.cc#L42>`_
.. [10] `<https://github.com/kaldi-asr/kaldi/blob/3b68c30991f8925b0973d1aeccfbd522be0748d3/src/feat/mel-computations.h#L85>`_
