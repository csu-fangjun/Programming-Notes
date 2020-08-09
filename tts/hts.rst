
HTS
===

.. code-block::

  # go to http://hts.sp.nitech.ac.jp/

  wget http://hts.sp.nitech.ac.jp/archives/2.3/HTS-2.3_for_HTK-3.4.1.tar.bz2
  mkdir hts
  cd hts
  tar xf HTS-2.3_for_HTK-3.4.1.tar.bz2
  cd ..
  git clone https://github.com/songmeixu/HTK.git
  cd HTK
  patch -p1 < ../hts/HTS-2.3_for_HTK-3.4.1.patch

  # read README for installation instructions

  ./configure --help
  ./configure --prefix=$HOME/software/htk
  make
  make install

  # LM tools
  make hlmtools
  make install-hlmtools

  # then install decoders
  make hdecode
  make install-hdecode

.. code-block::

    $ ls ~/software/htk/bin/
    Cluster  HCopy        HDMan   HInit  HLMCopy    HMGenS    HParse  HResults  HSMMAlign  LAdapt  LGCopy  LLink    LNorm
    HBuild   HDecode      HERest  HLEd   HLRescore  HMgeTool  HQuant  HSGen     HSmooth    LBuild  LGList  LMerge   LPlex
    HCompV   HDecode.mod  HHEd    HList  HLStats    HMMIRest  HRest   HSLab     HVite      LFoF    LGPrep  LNewMap  LSubset

Add ``$HOME/software/htk/bin`` to ``PATH``.


