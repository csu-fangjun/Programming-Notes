#!/usr/bin/env python3
#
# This file demonstrates how an arpa file is computed
# from ngram counts

import math


def get_unigram():
    # CAUTION: <s> is not used in compute prob for unigram
    unigram = {}
    unigram['<s>'] = 5
    unigram['hello'] = 3
    unigram['world'] = 4
    unigram['</s>'] = 5
    unigram['foo'] = 4
    unigram['bar'] = 3
    return unigram


def get_bigram():
    bigram = {}
    bigram['<s> hello'] = 1
    bigram['<s> foo'] = 3
    bigram['<s> world'] = 1
    bigram['hello world'] = 2
    bigram['hello </s>'] = 1
    bigram['world </s>'] = 2
    bigram['world bar'] = 2
    bigram['foo bar'] = 1
    bigram['foo </s>'] = 1
    bigram['foo hello'] = 1
    bigram['foo world'] = 1
    bigram['bar hello'] = 1
    bigram['bar foo'] = 1
    bigram['bar </s>'] = 1

    return bigram


def compute_unigram():
    print('\\1-grams:')
    unigram = get_unigram()
    # Note that <s> is not considered why computing `total`
    total = sum([count for key, count in unigram.items() if key != '<s>'])
    total = float(total)
    for key, count in unigram.items():
        if key == '<s>':
            print('-99 <s> -99')
            continue
        p = float(count) / total
        # NOTE: the first column is log10(p)
        print(math.log10(p), key, -99 if key != '</s>' else '')
    print()


def compute_bigram():
    print('\\2-grams:')
    unigram = get_unigram()
    bigram = get_bigram()

    for key, count in bigram.items():
        p = float(count) / unigram[key.split()[0]]
        print(math.log10(p), key)
    print()


def main():
    print('\data\\')
    unigram = get_unigram()
    bigram = get_bigram()
    print(f'ngram 1={len(unigram)}')
    print(f'ngram 2={len(bigram)}')
    print()
    compute_unigram()
    compute_bigram()

    print('\end\\')


if __name__ == '__main__':
    main()
