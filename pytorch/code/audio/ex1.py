#!/usr/bin/env  python3

import torchaudio


def main():
    #  print(torchaudio.__version__) # 0.8.0
    effect_names = torchaudio.sox_effects.effect_names()
    assert isinstance(effect_names, list)
    assert isinstance(effect_names[0], str)
    #  print(effect_names)
    # allpass, band, bandpass, bandreject,
    # bass, bend, biquad, chorus, channels,
    # compand, contrast, dcshift, deemph,
    # delay, dither, divide, downsample,
    # earwax, echo, echos, equalizer, fade,
    # fir, firfit, flanger, gain, ghpass,
    # hilbert, loudness, lowpass, mcompand,
    # norm, oops, overdrive, pad, phaser,
    # pitch, rate, remix, repeat, reverb,
    # reverse, riaa, silence, sinc, speed,
    # stat, stats, stretch, swap, synth,
    # tempo, treble, tremolo, trim, upsample,
    # vad, vol
    #  print(len(effect_names)) # 58


if __name__ == '__main__':
    main()
