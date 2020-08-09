
Monophone Training
==================

.. code-block::

    steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
      data/train_shorter data/lang exp/mono

Split data dir:

.. code-block::

    sdata=$data/split$nj;
    [[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

feats:

.. code-block::

  feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"
  example_feats="`echo $feats | sed s/JOB/1/g`";

- it first applies cmvn
- then it uses ``add-deltas`` (velocity and acceleration)

**data/lang**

  - oov.int
  - phones.txt
  - sets.int
  - topo

gmm-init-mono:

.. code-block::

      shared_phones_opt="--shared-phones=$lang/phones/sets.int"
      gmm-init-mono $shared_phones_opt "--train-feats=$feats subset-feats --n=10 ark:- ark:-|" $lang/topo $feat_dim \
      $dir/0.mdl $dir/tree || exit 1;

``subset-feats`` uses only the first 10 wavs.

The input are ``topo`` and ``feat-dim``, and the output are ``0.mdl`` and ``tree``.
``HmmTopology`` is used to read the ``topo`` file.

``tree`` is generated using ``ContextDependency``. A gmm with only one mixture component is
saved in ``AmDiagGmm``.

A ``TransitionModel`` consists of ``topo`` and ``tree`` (ContextDependency).
``0.mdl`` consists of a ``TransitionModel`` and a ``AmDiagGmm``.


.. code-block::

    numgauss=`gmm-info --print-args=false $dir/0.mdl | grep gaussians | awk '{print $NF}'`
    incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss

By default, ``totgauss`` is 1000.  ``max_iter_inc`` is 30.

gmm-info:

It uses a ``TransitionModel`` and a ``AmDiagGmm`` to read the file. It prints:

- number of phones, from transition model
- number of pdfs, from transition model
- number of transition ids. from transition model
- number of transition states, from transition model
- feature dim, from AmDiagGmm
- number of gaussians, from AmDiagGmm, (sum of number of mixtures of all pdfs)


.. code-block::

    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/0.mdl  $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;

``compile-train-graphs``
- input: tree, 0.mdl, L.fst, transcript_rspecifier, 
- output: fst_wspecifier, a table of HCLG.fst, the input is transition id and the output is word.

It uses ``ContextDependency`` to read the tree. ``TransitionModel`` is used to read ``0.mdl``.
Only the transition model in ``0.mdl`` is used; its ``AmDiagGmm`` is not used.


.. code-block::

    bool TrainingGraphCompiler::CompileGraphFromText(
        const std::vector<int32> &transcript,
        fst::VectorFst<fst::StdArc> *out_fst) {
      using namespace fst;
      VectorFst<StdArc> word_fst;
      MakeLinearAcceptor(transcript, &word_fst);
      return CompileGraph(word_fst, out_fst);
    }

``MakeLinearAcceptor`` is in ``fstext/fstext-utils-inl.h``.


.. code-block::

    VectorFst<StdArc> phone2word_fst;
    // TableCompose more efficient than compose.
    TableCompose(*lex_fst_, word_fst, &phone2word_fst, &lex_cache_);

    KALDI_ASSERT(phone2word_fst.Start() != kNoStateId);

``TableCompose`` is defined in ``fstext/table-matcher.h``.

.. code-block::

  const std::vector<int32> &phone_syms = trans_model_.GetPhones();  // needed to create context fst.

  // inv_cfst will be expanded on the fly, as needed.
  InverseContextFst inv_cfst(subsequential_symbol_,
                             phone_syms,
                             disambig_syms_,
                             ctx_dep_.ContextWidth(),
                             ctx_dep_.CentralPosition());

``InverseContextFst`` is in ``fstext/context-fst.cc``

InverseContextFst transduces phones to context windows.


.. code-block::

  std::vector<std::vector<int32> > ilabel_info_;

It is indexed by fst input label, in the case of InverseContextFst, it is phone id.
It saves the corresponding context window.

For example, suppose symbol 1500 is phone 30 with a right-context of 12 and a left-context of 4, we would have::

  // not valid C++
  ilabel_info[1500] == { 4, 30, 12 };

.. code-block::

  VectorFst<StdArc> ctx2word_fst;
  ComposeDeterministicOnDemandInverse(phone2word_fst, &inv_cfst, &ctx2word_fst);
  // now ctx2word_fst is C * LG, assuming phone2word_fst is written as LG.
  KALDI_ASSERT(ctx2word_fst.Start() != kNoStateId);

.. code-block::

  HTransducerConfig h_cfg;
  h_cfg.transition_scale = opts_.transition_scale;

  std::vector<int32> disambig_syms_h; // disambiguation symbols on
  // input side of H.
  VectorFst<StdArc> *H = GetHTransducer(inv_cfst.IlabelInfo(),
                                        ctx_dep_,
                                        trans_model_,
                                        h_cfg,
                                        &disambig_syms_h);

  VectorFst<StdArc> &trans2word_fst = *out_fst;  // transition-id to word.
  TableCompose(*H, ctx2word_fst, &trans2word_fst);

The input of ``H.fst`` is transition id; its output is the index into ``ilabel_info``,
which contains a context window of phones, which is used as input to ``C.fst``.

.. code-block::

  echo "$0: Aligning data equally (pass 0)"
  $cmd JOB=1:$nj $dir/log/align.0.JOB.log \
    align-equal-compiled "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,t:-  \| \
    gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
    $dir/0.JOB.acc || exit 1;

align-equal-compiled:

- input: fst_rspecifier, feats.scp
- output: alignment_wspecifier
- Note that it does not need a transition model.

.. code-block::

      // source-code for align-equal-compiled
      Int32VectorWriter alignment_writer(alignment_wspecifier);
      if (EqualAlign(decode_fst, features.NumRows(), rand_seed, &path) ) {
        std::vector<int32> aligned_seq, words;
        StdArc::Weight w;
        GetLinearSymbolSequence(path, &aligned_seq, &words, &w);
        KALDI_ASSERT(aligned_seq.size() == features.NumRows());
        alignment_writer.Write(key, aligned_seq);
        done++;
      } else {

``aligned_seq`` is a list of transition ids.

``gmm-acc-stats-ali``:

.. code-block::

    AmDiagGmm am_gmm;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_gmm.Read(ki.Stream(), binary);
    }

.. code-block::

    if [ $stage -le 0 ]; then
      gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
        $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
      rm $dir/0.*.acc
    fi

``gmm-sum-accs``: combines the output of ``gmm-acc-stats-ali`` into a single file.

``gmm-est``:

.. code-block::

    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "$0: Aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam --careful=$careful "$mdl" \
        "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" \
        || exit 1;
    fi

gmm-align-compiled:

.. code-block::

        DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                               acoustic_scale);
        AlignUtteranceWrapper(align_config, utt,
                              acoustic_scale, &decode_fst, &gmm_decodable,
                              &alignment_writer, &scores_writer,
                              &num_done, &num_err, &num_retry,
                              &tot_like, &frame_count, &per_frame_acwt_writer);


The output of ``gmm-align-compiled`` is ``*.ali``.

.. code-block::

    steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
      exp/mono/graph data/dev exp/mono/decode_dev

.. code-block::

    $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
      gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
      --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt $decode_extra_opts \
      $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

.. code-block::

    LatticeFasterDecoder decoder(*decode_fst, config);

    DecodableAmDiagGmmScaled gmm_decodable(am_gmm, trans_model, features,
                                           acoustic_scale);

    if (DecodeUtteranceLatticeFaster(
            decoder, gmm_decodable, trans_model, word_syms, utt,
            acoustic_scale, determinize, allow_partial, &alignment_writer,
            &words_writer, &compact_lattice_writer, &lattice_writer,
            &like)) {

The output of ``gmm-latgen-faster`` is ``lat.*.gz``


steps/train_deltas.sh
---------------------

.. code-block::

    if [ $stage -le -3 ]; then
      echo "$0: accumulating tree stats"
      $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
        acc-tree-stats $context_opts \
        --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
        "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
      sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
      rm $dir/*.treeacc
    fi

``--ci-phones``, it is the colon ``:`` separated phone ids for silence phones;
only the transition model is read from ``final.mdl``.

.. code-block::

    std::map<EventType, GaussClusterable*> tree_stats;

    AccumulateTreeStats(trans_model,
                        acc_tree_stats_info,
                        alignment,
                        mat,
                        &tree_stats);

