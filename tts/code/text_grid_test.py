#!/usr/bin/env python3

# xxx.TextGrid can be generated with `mfa_align`
# The inputs of mfa_align are:
#  1. a directory containing a list of wav files
#  2. a list of *.txt file corresponding to the wav files; they live
#     in the same directory. The *.txt are the transcritps of the wavs.
#  3. A dictionary
#  4. a pretrained model

# pip install tgt

import tgt
import os

s = r'''
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 1.170
tiers? <exists>
size = 2
item []:
	item [1]:
		class = "IntervalTier"
		name = "words"
		xmin = 0.0
		xmax = 1.170
		intervals: size = 4
			intervals [1]:
				xmin = 0.000
				xmax = 0.670
				text = "printing"
			intervals [2]:
				xmin = 0.670
				xmax = 0.800
				text = ""
			intervals [3]:
				xmin = 0.800
				xmax = 1.000
				text = "in"
			intervals [4]:
				xmin = 1.000
				xmax = 1.170
				text = "the"
	item [2]:
		class = "IntervalTier"
		name = "phones"
		xmin = 0.0
		xmax = 0.670
		intervals: size = 7
			intervals [1]:
				xmin = 0.000
				xmax = 0.040
				text = "P"
			intervals [2]:
				xmin = 0.040
				xmax = 0.070
				text = "R"
			intervals [3]:
				xmin = 0.070
				xmax = 0.190
				text = "IH1"
			intervals [4]:
				xmin = 0.190
				xmax = 0.220
				text = "N"
			intervals [5]:
				xmin = 0.220
				xmax = 0.300
				text = "T"
			intervals [6]:
				xmin = 0.300
				xmax = 0.450
				text = "IH0"
			intervals [7]:
				xmin = 0.450
				xmax = 0.670
				text = "NG"
'''


def main():
    filename = 'a.txt'
    with open(filename, 'w') as f:
        f.write(s)
    textgrid = tgt.io.read_textgrid(filename)
    assert isinstance(textgrid, tgt.core.TextGrid)

    words_interval_tier = textgrid.get_tier_by_name('words')
    # print(interval_tier)
    # IntervalTier(start_time=0.0, end_time=1.17, name="words"), objects=[
    # Interval(0.0, 0.67, "printing"),
    # Interval(0.8, 1.0, "in"),
    # Interval(1.0, 1.17, "the")])

    phones_interval_tier = textgrid.get_tier_by_name('phones')
    #  print(phones_interval_tier)
    # IntervalTier(start_time=0.0, end_time=0.67, name="phones", objects=[
    # Interval(0.0, 0.04, "P"),
    # Interval(0.04, 0.07, "R"),
    # Interval(0.07, 0.19, "IH1"),
    # Interval(0.19, 0.22, "N"),
    # Interval(0.22, 0.3, "T"),
    # Interval(0.3, 0.45, "IH0"),
    # Interval(0.45, 0.67, "NG")
    # ])
    assert phones_interval_tier.start_time == 0.0
    assert phones_interval_tier.end_time == 0.67
    assert phones_interval_tier.name == 'phones'

    for t in phones_interval_tier:
        assert isinstance(t, tgt.core.Interval)
        assert hasattr(t, 'start_time')
        assert hasattr(t, 'end_time')
        assert hasattr(t, 'text')

    os.remove(filename)


if __name__ == '__main__':
    main()
