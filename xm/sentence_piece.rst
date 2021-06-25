SentencePiece
=============

Python
------

.. code-block::

  pip install sentencepiece

C++
---

.. code-block::

  git clone https://github.com/google/sentencepiece.git
  cd sentencepiece
  mkdir build
  cd build
  cmake -DCMAKE_BUILD_TYPE=Release ..
  make -j

  # add $PWD/build/src to $PATH

.. code-block::

  fangjun_s2:~/open-source/sentencepiece/build$ ls -lh ./src/
  total 7.1M
  drwxr-xr-x 11 kuangfangjun root 4.0K Jun 17 15:15 CMakeFiles
  -rw-r--r--  1 kuangfangjun root 116K Jun 17 15:15 Makefile
  -rw-r--r--  1 kuangfangjun root 9.5K Jun 17 15:15 cmake_install.cmake
  -rw-r--r--  1 kuangfangjun root 2.1M Jun 17 15:16 libsentencepiece.a
  lrwxrwxrwx  1 kuangfangjun root   21 Jun 17 15:16 libsentencepiece.so -> libsentencepiece.so.0
  lrwxrwxrwx  1 kuangfangjun root   25 Jun 17 15:16 libsentencepiece.so.0 -> libsentencepiece.so.0.0.0
  -rwxr-xr-x  1 kuangfangjun root 1.3M Jun 17 15:16 libsentencepiece.so.0.0.0
  -rw-r--r--  1 kuangfangjun root 2.0M Jun 17 15:16 libsentencepiece_train.a
  lrwxrwxrwx  1 kuangfangjun root   27 Jun 17 15:16 libsentencepiece_train.so -> libsentencepiece_train.so.0
  lrwxrwxrwx  1 kuangfangjun root   31 Jun 17 15:16 libsentencepiece_train.so.0 -> libsentencepiece_train.so.0.0.0
  -rwxr-xr-x  1 kuangfangjun root 1.6M Jun 17 15:16 libsentencepiece_train.so.0.0.0
  -rwxr-xr-x  1 kuangfangjun root  40K Jun 17 15:16 spm_decode
  -rwxr-xr-x  1 kuangfangjun root  76K Jun 17 15:16 spm_encode
  -rwxr-xr-x  1 kuangfangjun root  25K Jun 17 15:16 spm_export_vocab
  -rwxr-xr-x  1 kuangfangjun root  34K Jun 17 15:16 spm_normalize
  -rwxr-xr-x  1 kuangfangjun root  45K Jun 17 15:16 spm_train

.. code-block::

  $ spm_train --input=data/botchan.txt --vocab_size=5000 --model_type=unigram --model_prefix=me_5000

  sentencepiece_trainer.cc(77) LOG(INFO) Starts training with :
  trainer_spec {
    input: data/botchan.txt
    input_format:
    model_prefix: me_5000
    model_type: UNIGRAM
    vocab_size: 5000
    self_test_sample_size: 0
    character_coverage: 0.9995
    input_sentence_size: 0
    shuffle_input_sentence: 1
    seed_sentencepiece_size: 1000000
    shrinking_factor: 0.75
    max_sentence_length: 4192
    num_threads: 16
    num_sub_iterations: 2
    max_sentencepiece_length: 16
    split_by_unicode_script: 1
    split_by_number: 1
    split_by_whitespace: 1
    split_digits: 0
    treat_whitespace_as_suffix: 0
    allow_whitespace_only_pieces: 0
    required_chars:
    byte_fallback: 0
    vocabulary_output_piece_score: 1
    train_extremely_large_corpus: 0
    hard_vocab_limit: 1
    use_all_vocab: 0
    unk_id: 0
    bos_id: 1
    eos_id: 2
    pad_id: -1
    unk_piece: <unk>
    bos_piece: <s>
    eos_piece: </s>
    pad_piece: <pad>
    unk_surface:  _
  }
  denormalizer_spec {}
  trainer_interface.cc(329) LOG(INFO) SentenceIterator is not specified. Using MultiFileSentenceIterator.
  trainer_interface.cc(178) LOG(INFO) Loading corpus: data/botchan.txt
  trainer_interface.cc(385) LOG(INFO) Loaded all 4288 sentences
  trainer_interface.cc(400) LOG(INFO) Adding meta_piece: <unk>
  trainer_interface.cc(400) LOG(INFO) Adding meta_piece: <s>
  trainer_interface.cc(400) LOG(INFO) Adding meta_piece: </s>
  trainer_interface.cc(405) LOG(INFO) Normalizing sentences...
  trainer_interface.cc(466) LOG(INFO) all chars count=274252
  trainer_interface.cc(477) LOG(INFO) Done: 99.957% characters are covered.
  trainer_interface.cc(487) LOG(INFO) Alphabet size=69
  trainer_interface.cc(488) LOG(INFO) Final character coverage=0.99957
  trainer_interface.cc(520) LOG(INFO) Done! preprocessed 4288 sentences.
  unigram_model_trainer.cc(139) LOG(INFO) Making suffix array...
  unigram_model_trainer.cc(143) LOG(INFO) Extracting frequent sub strings...
  unigram_model_trainer.cc(194) LOG(INFO) Initialized 16112 seed sentencepieces
  trainer_interface.cc(526) LOG(INFO) Tokenizing input sentences with whitespace: 4288
  trainer_interface.cc(537) LOG(INFO) Done! 9165
  unigram_model_trainer.cc(489) LOG(INFO) Using 9165 sentences for EM training
  unigram_model_trainer.cc(505) LOG(INFO) EM sub_iter=0 size=5926 obj=10.5283 num_tokens=18931 num_tokens/piece=3.19457
  unigram_model_trainer.cc(505) LOG(INFO) EM sub_iter=1 size=5232 obj=8.64492 num_tokens=19009 num_tokens/piece=3.63322
  trainer_interface.cc(615) LOG(INFO) Saving model: me_5000.model
  trainer_interface.cc(626) LOG(INFO) Saving vocabs: me_5000.vocab

.. code-block::

  $ echo "hello world" > a.txt
  $ spm_encode --model ./me_100.model  --output_format=piece < ./a.txt
  _ h e ll o _w or l d
  $ spm_encode --model ./me_100.model  --output_format=piece < ./a.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}'

  d 2
  e 3
  h 4
  l 5
  ll 6
  o 7
  or 8
  _ 9
  _w 10

  # The above output is appended to a dict file, which intially has one line "<unk> 1"

.. code-block::

  $ spm_encode  --help
  sentencepiece

  Usage: spm_encode [options] files

     --help (show help)  type: bool default: false
     --version (show version)  type: bool default: false
     --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0
     --model (model file name)  type: std::string default: ""
     --output_format (choose from piece, id, proto, nbest_piece, nbest_id, or nbest_proto)  type: std::string default: "piece"
     --input (input filename)  type: std::string default: ""
     --output (output filename)  type: std::string default: ""
     --extra_options (':' separated encoder extra options, e.g., "reverse:bos:eos")  type: std::string default: ""
     --nbest_size (NBest size)  type: int32 default: 10
     --alpha (Smoothing parameter for sampling mode.)  type: double default: 0.5
     --random_seed (Seed value for random generator.)  type: uint32 default: 4294967295
     --vocabulary (Restrict the vocabulary. The encoder only emits the tokens in "vocabulary" file)  type: std::string default: ""
     --vocabulary_threshold (Words with frequency < threshold will be treated as OOV)  type: int32 default: 0
     --generate_vocabulary (Generates vocabulary file instead of segmentation)  type: bool default: false

.. code-block::

  $ spm_train --help
  sentencepiece

  Usage: spm_train [options] files

     --help (show help)  type: bool default: false
     --version (show version)  type: bool default: false
     --minloglevel (Messages logged at a lower level than this don't actually get logged anywhere)  type: int default: 0
     --input (comma separated list of input sentences)  type: std::string default: ""
     --input_format (Input format. Supported format is `text` or `tsv`.)  type: std::string default: ""
     --model_prefix (output model prefix)  type: std::string default: ""
     --model_type (model algorithm: unigram, bpe, word or char)  type: std::string default: "unigram"
     --vocab_size (vocabulary size)  type: int32 default: 8000
     --accept_language (comma-separated list of languages this model can accept)  type: std::string default: ""
     --self_test_sample_size (the size of self test samples)  type: int32 default: 0
     --character_coverage (character coverage to determine the minimum symbols)  type: double default: 0.9995
     --input_sentence_size (maximum size of sentences the trainer loads)  type: std::uint64_t default: 0
     --shuffle_input_sentence (Randomly sample input sentences in advance. Valid when --input_sentence_size > 0)  type: bool default: true
     --seed_sentencepiece_size (the size of seed sentencepieces)  type: int32 default: 1000000
     --shrinking_factor (Keeps top shrinking_factor pieces with respect to the loss)  type: double default: 0.75
     --num_threads (number of threads for training)  type: int32 default: 16
     --num_sub_iterations (number of EM sub-iterations)  type: int32 default: 2
     --max_sentencepiece_length (maximum length of sentence piece)  type: int32 default: 16
     --max_sentence_length (maximum length of sentence in byte)  type: int32 default: 4192
     --split_by_unicode_script (use Unicode script to split sentence pieces)  type: bool default: true
     --split_by_number (split tokens by numbers (0-9))  type: bool default: true
     --split_by_whitespace (use a white space to split sentence pieces)  type: bool default: true
     --split_digits (split all digits (0-9) into separate pieces)  type: bool default: false
     --treat_whitespace_as_suffix (treat whitespace marker as suffix instead of prefix.)  type: bool default: false
     --allow_whitespace_only_pieces (allow pieces that only contain (consecutive) whitespace tokens)  type: bool default: false
     --control_symbols (comma separated list of control symbols)  type: std::string default: ""
     --control_symbols_file (load control_symbols from file.)  type: std::string default: ""
     --user_defined_symbols (comma separated list of user defined symbols)  type: std::string default: ""
     --user_defined_symbols_file (load user_defined_symbols from file.)  type: std::string default: ""
     --required_chars (UTF8 characters in this flag are always used in the character set regardless of --character_coverage)  type: std::string default: ""
     --byte_fallback (decompose unknown pieces into UTF-8 byte pieces)  type: bool default: false
     --vocabulary_output_piece_score (Define score in vocab file)  type: bool default: true
     --normalization_rule_name (Normalization rule name. Choose from nfkc or identity)  type: std::string default: "nmt_nfkc"
     --normalization_rule_tsv (Normalization rule TSV file. )  type: std::string default: ""
     --denormalization_rule_tsv (Denormalization rule TSV file.)  type: std::string default: ""
     --add_dummy_prefix (Add dummy whitespace at the beginning of text)  type: bool default: true
     --remove_extra_whitespaces (Removes leading, trailing, and duplicate internal whitespace)  type: bool default: true
     --hard_vocab_limit (If set to false, --vocab_size is considered as a soft limit.)  type: bool default: true
     --use_all_vocab (If set to true, use all tokens as vocab. Valid for word/char models.)  type: bool default: false
     --unk_id (Override UNK (<unk>) id.)  type: int32 default: 0
     --bos_id (Override BOS (<s>) id. Set -1 to disable BOS.)  type: int32 default: 1
     --eos_id (Override EOS (</s>) id. Set -1 to disable EOS.)  type: int32 default: 2
     --pad_id (Override PAD (<pad>) id. Set -1 to disable PAD.)  type: int32 default: -1
     --unk_piece (Override UNK (<unk>) piece.)  type: std::string default: "<unk>"
     --bos_piece (Override BOS (<s>) piece.)  type: std::string default: "<s>"
     --eos_piece (Override EOS (</s>) piece.)  type: std::string default: "</s>"
     --pad_piece (Override PAD (<pad>) piece.)  type: std::string default: "<pad>"
     --unk_surface (Dummy surface string for <unk>. In decoding <unk> is decoded to `unk_surface`.)  type: std::string default: " _ "
     --train_extremely_large_corpus (Increase bit depth for unigram tokenization.)  type: bool default: false
     --random_seed (Seed value for random generator.)  type: uint32 default: 4294967295

