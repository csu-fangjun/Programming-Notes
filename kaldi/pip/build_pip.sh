#!/bin/bash

KALDI_ROOT=/path/to/kaldi

package_dir=$PWD/kaldi
mkdir -p $package_dir

kaldi_pybind_so=kaldi_pybind.cpython-35m-x86_64-linux-gnu.so
src_lib_dir=$KALDI_ROOT/src/lib
dst_lib_dir=$package_dir/lib

mkdir -p $dst_lib_dir

so_files=(
libkaldi-base.so
libkaldi-chain.so
libkaldi-cudamatrix.so
libkaldi-decoder.so
libkaldi-feat.so
libkaldi-fstext.so
libkaldi-hmm.so
libkaldi-lat.so
libkaldi-matrix.so
libkaldi-nnet3.so
libkaldi-util.so
)

for f in ${so_files[@]}; do
  if [[ ! -e $dst_lib_dir/$f ]]; then
    cp -v $src_lib_dir/$f $dst_lib_dir/$f
  fi
done

if [[ ! -f $dst_lib_dir/libwarpctc.so ]]; then
  cp $KALDI_ROOT/src/pybind/ctc/warp-ctc/build/libwarpctc.so $dst_lib_dir
fi

fst_so_files=(
libfstscript.so.10
libfst.so.10
)

fst_lib_dir=$KALDI_ROOT/tools/openfst/lib

for f in ${fst_so_files[@]}; do
  if [[ ! -f $dst_lib_dir/$f ]]; then
    cp -v $fst_lib_dir/$f $dst_lib_dir/$f
  fi
done


cp -v $KALDI_ROOT/src/pybind/kaldi/*.py $package_dir

if [[ ! -f $package_dir/$kaldi_pybind_so ]]; then
  cp -v $KALDI_ROOT/src/pybind/$kaldi_pybind_so $package_dir
fi

chrpath -r '$ORIGIN/lib' $package_dir/$kaldi_pybind_so

python3 setup.py bdist_wheel
