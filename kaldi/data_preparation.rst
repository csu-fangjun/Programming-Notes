
Data Preparation
================

**text**
  We need to generate a file with containing lines of ``key transcript``

**utt2spk**
  Utterance to speaker

**spk2utt**
  Speaker to Utterance

**wav.scp**
  A file containing ``key /path/to.wav``

  The value can also be something like ``sox /path/to.wav -t wav - |``,
  where ``-t wav`` says the input format is ``wav``.



Useful tools for data preparation
---------------------------------

awk
^^^

**awk**
  .. code-block::

    echo "123.wav hello" | awk '{split($1, a, "."); print a[1]}'

  ``a[1]`` prints ``123``; ``a[2]`` prints ``wav``.

  .. code-block::

    paste -d " " <(echo "123.wav hello" | awk '{split($1, a, "."); print a[1]}') <(echo hello)

  The output is ``123 hello``

sort
^^^^

**sort**
  .. code-block::

    (echo "1 hello"; echo "3 world"; echo "20 foo"; echo "1 hello") | sort
    (echo "1 hello"; echo "3 world"; echo "20 foo"; echo "1 hello") | sort -u
    (echo "1 hello"; echo "3 world"; echo "20 foo"; echo "1 hello") | sort -u -n
    (echo "1 hello"; echo "3 world"; echo "20 foo"; echo "1 hello") | sort -u -k2
    (echo "1 hello"; echo "3 world"; echo "20 foo"; echo "1 hello") | sort -u -k2,2

  ``-u`` means ``unique``; ``-n`` is for numerical comparison; ``-k`` specifies the field.
  ``-k2,2`` specifies both the start field and the end field.

filter_scp.pl
^^^^^^^^^^^^^

**utils/filter_scp.pl**
  - ``utils/filter_scp.pl --include id_mask_list in.scp > out.scp``. (intersection)
  - ``utils/filter_scp.pl --exclude id_mask_list in.scp > out.scp``  (difference, out = in.scp - id_mask_list)
  - ``cat in.scp | utils/filter_scp.pl --include id_mask_list > out.scp``
  - See the help information in the script file.

  .. code-block::

    utils/filter_scp.pl <(awk '{if ($2 == "FREETEXT") print $1}' ${srcdir}_whole/text) \
      ${srcdir}_whole/segments >${srcdir}_whole/neg_segments

    utils/filter_scp.pl --exclude ${srcdir}_whole/neg_segments ${srcdir}_whole/segments \
      >${srcdir}_whole/pos_segments

    utils/filter_scp.pl ${srcdir}_whole/pos_segments ${srcdir}_whole/utt2dur >${srcdir}_whole/pos_utt2dur

fix_data_dir.sh
^^^^^^^^^^^^^^^

**utils/fix_data_dir.sh**
  - ``utils/fix_data_dir.sh in_out_data_dir``; ``in_out_data_dir`` have to contain ``text``, ``wav.scp`` and ``utt2spk`` before invoking this script
  - It checks that files (e.g., ``utt2spk``) are sorted and unique; if not, then sort them in-place. It uses ``sort -k1,1 -u``
  - It will generate ``spk2utt`` if it does not exist.
  - It is usually invoked before and after generating features.

  .. code-block::

    utils/fix_data_dir.sh data
    steps/make_mfcc.sh data
    steps/compute_cmvn_stats.sh data
    utils/fix_data_dir.sh data

subset_data_dir.sh
^^^^^^^^^^^^^^^^^^

  **utils/subset_data_dir.sh**

    .. code-block::

      grep "music" ${data_dir}/musan/utt2spk > local/musan.tmp/utt2spk_music

      utils/subset_data_dir.sh --utt-list local/musan.tmp/utt2spk_music \
              ${data_dir}/musan ${data_dir}/musan_music

      utils/fix_data_dir.sh ${data_dir}/musan_music



validate_data_dir.sh
^^^^^^^^^^^^^^^^^^^^

**utils/validate_data_dir.sh**
  It can be used after generating features.

  .. code-block::

      for name in reverb noise music babble; do
        steps/make_mfcc.sh --nj 16 --cmd "$train_cmd" \
          data/train_shorter_${name} || exit 1;
        steps/compute_cmvn_stats.sh data/train_shorter_${name}
        utils/fix_data_dir.sh data/train_shorter_${name}
        utils/validate_data_dir.sh data/train_shorter_${name}
      done


split_scp.pl
^^^^^^^^^^^^

**utils/split_scp.pl**
  It is used to generate ``1.scp``, ``2.scp``, ....

  Common usages is::

    split_scps=
    for n in $(seq $nj); do
      split_scps="$split_scps $logdir/wav_${name}.$n.scp"
    done

    utils/split_scp.pl $scp $split_scps || exit 1;

  Note that it counts from ``1``.

split_data.sh
^^^^^^^^^^^^^

  **utils/data/split_data.sh**

.. code-block::

    utils/data/split_data.sh data 10


It creates ``data/split10/1``, ``data/split10/2``, ... Each splitted directory contains the same
files as ``data``, but the number of lines in each file is reduced. By default, all utterances
of a speaker resides in a single split directory.

.. code-block::

    utils/data/split_data.sh --per-utt data 10
    sdata=data/split10utt

When ``--per-utt`` is passed, it creates ``data/split10utt/1``, ``data/split10utt/2``, ...

make_mfcc.sh
^^^^^^^^^^^^

**steps/make_mfcc.sh**
  The default config file is ``conf/mfcc.conf``. The file has to be exist, even if it is empty.

  .. code-block::

    steps/make_mfcc.sh data
    steps/make_mfcc.sh data data/log
    steps/make_mfcc.sh data data/log data/data

  The script has a default option ``write_utt2num_frames=true``, which is passed as option ``--write-num-frames=ark,t:$logdir/utt2num_frames.JOB``
  to  ``copy-feats``. It will generate a file named ``utt2num_frames`` with the following format::

    key1 100
    key2 200

  where ``100`` and ``200`` are number of frames. The file is created with ``Int32Writer``.

  Another default option of the script is ``write_utt2dur=true``, which is passed as option
  ``--write-utt2dur=ark,t:$logdir/utt2dur.JOB`` to ``compute-mfcc-feats``. It creates a file
  named ``utt2dur`` with the following format::

    key1 1.606
    key2 1.717

  where ``1.606`` and ``1.717`` are number of seconds. It is created using ``DoubleWriter``.

  It also generates a file ``frame_shift``, which usually contains ``0.01``. Thata is, 0.01 seconds, or ``10 ms``.

  The config file is copied to ``data/conf/mfcc.conf``.

  In summary, after ``make_mfcc.sh``, we have the following files:

    - ``feats.scp``
    - ``frame_shift``
    - ``conf/mfcc.conf``
    - ``utt2dur`` if ``write_utt2dur=true``, which is default to ``true``
    - ``utt2num_frames`` if ``write_utt2num_frames=true``, which is default to ``true``


prepare_lang.sh
^^^^^^^^^^^^^^^

**utils/prepare_lang.sh**
  It requires the following input files:

    **nonsilence_phones.txt**
      Example format::

        a
        b

    **silence_phones.txt**
      Example format::

        <SIL>

    **optional_silence.txt**
      Example format::

        <SIL>

    **lexicon.txt**
      Example format::

        hello a
        world b
        sil   <SIL>

    **extra_questions.txt**
      It can be empty.

  Example usage::

    utils/prepare_lang.sh --num-sil-states 1 --num-nonsil-states 4 --sil-prob 0.5 \
      --position-dependent-phones false \
      data/local/dict "<sil>" data/lang/temp data/lang

  It will generate ``topo``, ``L.fst``, ``phones.txt``, etc.

validate_lang.pl
^^^^^^^^^^^^^^^^

  **utils/validate_lang.pl**
    ``utils/validate_lang.pl data/lang``


copy_data_dir.sh
^^^^^^^^^^^^^^^^
  **utils/copy_data_dir.sh**
    It copies ``feats.scp``, ``utt2spk``, ``spk2utt``, ``wav.scp``, ``text``, ``frame_shift``,
    ``utt2dur``, ``utt2num_frames``, etc, from source dir to dest dir.

    We can add prefix and suffix to the utt id and spk id.


    Note that it copies only text files. No ark files is copied.

apply_map.pl
^^^^^^^^^^^^

  **utils/apply_map.pl**
    The following example code is copied from ``utils/copy_data_dir.sh``::

        cat $srcdir/utt2spk | awk -v p=$utt_prefix -v s=$utt_suffix '{printf("%s %s%s%s\n", $1, p, $1, s);}' > $destdir/utt_map
        cat $srcdir/spk2utt | awk -v p=$spk_prefix -v s=$spk_suffix '{printf("%s %s%s%s\n", $1, p, $1, s);}' > $destdir/spk_map

        cat $srcdir/utt2spk | utils/apply_map.pl -f 1 $destdir/utt_map  | \
          utils/apply_map.pl -f 2 $destdir/spk_map >$destdir/utt2spk

        utils/utt2spk_to_spk2utt.pl <$destdir/utt2spk >$destdir/spk2utt

        if [ -f $srcdir/feats.scp ]; then
          utils/apply_map.pl -f 1 $destdir/utt_map <$srcdir/feats.scp >$destdir/feats.scp
        fi


  .. code-block::

      # Create a mapping from the new to old utterances.  This file will be deleted later.
      awk '{print $1, $2}' < $subsegments > $dir/new2old_utt

      # Create the new utt2spk file [just map from the second field
      utils/apply_map.pl -f 2 $srcdir/utt2spk < $dir/new2old_utt >$dir/utt2spk

      awk '{print $1,$2}' ${srcdir}_whole/sub_segments | \
        utils/apply_map.pl -f 2 ${srcdir}_whole/text >data/train_segmented/text


convert_data_dir_to_whole.sh
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  **utils/data/convert_data_dir_to_whole.sh**


get_utt2dur.sh
^^^^^^^^^^^^^^

  **utils/data/get_utt2dur.sh**
    - if the file `data/utt2dur` exists, then skip.
    - if `data/frame_shift` and `data/utt2num_frames` exist, then compute `utt2dur` from them
    - if `data/wav.scp` exists, then use `featbin/wav-to-duration.cc` to compute it. ``utils/data/split_data.sh`` is used for ``run.pl``.
    - if ``data/feats.scp`` exists, it uses ``feat-to-len`` with default frame shift ``0.01`` seconds.

    .. code-block::

        # The 1.5 correction is the typical value of (frame_length-frame_shift)/frame_shift.
        feat-to-len scp:$data/feats.scp ark,t:- |
          awk -v frame_shift=$frame_shift '{print $1, ($2+1.5)*frame_shift}' >$data/utt2dur

get_segments_for_data.sh
^^^^^^^^^^^^^^^^^^^^^^^^^

  **utils/data/get_segments_for_data.sh**
    .. code-block::

      # <utt-id> <utt-id> 0 <utt-dur>
      awk '{ print $1, $1, 0, $2 }' $data/utt2dur

    ``utils/data/get_segments_for_data.sh data > data/segments``

    Note that a line in a `segments` file has four fields.

subsegment_data_dir.sh
^^^^^^^^^^^^^^^^^^^^^^

  **utils/data/subsegment_data_dir.sh**
    .. code-block::

      utils/data/subsegment_data_dir.sh ${srcdir}_whole \
        ${srcdir}_whole/sub_segments data/train_segmented

extract-segments
^^^^^^^^^^^^^^^^

  **src/featbin/extract-segments.cc**
    Create a new ``wav.scp`` from an old ``wav.scp`` and ``segments`` file. It main purpose it to discard ``segments``.
    With ``wav.scp``, we can compute ``cmvn``.

    .. code-block::

      utils/data/extract_wav_segments_data_dir.sh --nj 50 --cmd "$train_cmd" \
        data/train_segmented data/train_shorter
      steps/compute_cmvn_stats.sh data/train_shorter
      utils/fix_data_dir.sh data/train_shorter
      utils/validate_data_dir.sh data/train_shorter

wav-reverberate
^^^^^^^^^^^^^^^

  **src/featbin/wav-reverberate.cc**

musan
^^^^^

  **steps/data/make_musan.sh**

    .. code-block::

      steps/data/make_musan.sh /path/src/musan data

    It invokes ``steps/data/make_musan.py`` to generate ``utt2spk``, ``wav.scp`` and ``spk2utt`.
    Note that there is no ``text`` file.


Summary
-------

1. Prepare ``text`` and ``wav.scp``, ``utt2spk``. If negative samples are tool long, we can break it
   into segments.
2. Add noise. There are two types of noise:

    - reverberation: through convolution, RIRS.zip (root impulse response)
    - additive noise: musan

