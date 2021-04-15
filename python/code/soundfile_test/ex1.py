#!/usr/bin/env python3

import soundfile


def main():
    filename = './stereo.sph'
    sf = soundfile.info(filename)
    print(sf)


if __name__ == '__main__':
    main()
