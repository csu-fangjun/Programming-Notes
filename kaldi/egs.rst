
egs
===


1. speed perturb, make mfcc, alignment, lattices
2. generate chain_topo,
3. build tree for chain, the alignment is converted to this tree.
4. estimate phone lm, its inputs are the chain_tree/alignment.
5. generate den.fst, normalization.fst from chain/tree, chain/0.trans_mdl, phone_lm.fst
6. volume perturb, make hires mfcc,
7. generate egs from hires mfcc

steps/nnet3/chain/get_egs.sh



- cmvn_opts: --norm-means false --norm-vars false
- frame subsampling factor: 3
- frames-overlap-per-eg: 0
- frames-per-eg: 150,110,90
- frames-per-iter: 1500000
- generate-egs-scp: true
- left-context: model-left-context + 1
- left-context-initial: -1
- left-tolerance: 5
- right-context: model-right-context + 1
- right-context-final: -1
- right-tolerance: 5
- srand: 0
- stage: -10
- data_dir: data/train_sp_hires, it contains: feats.scp
- chain_dir: exp/chain, it contains:  0.trans_mdl, tree, normalization.fst
- lat_dir: generated from the low resolution training data (speed perturbed), it contains: lat.1.gz
- egs_dir:exp/chain/egs


We need to know:

- frame shift: 0.01
- num_pdfs: from the tree using tree-info
- feat dim: from feat-to-dim

num_archives: num_frames / frames_per_iter + 1
egs_per_archive:  num_frames / (frames_per_eg * num_archives)


lattice-align-phones:

- replace-output-symbols: true
- output is phone lattices.

chain-get-supervision:

- lattice-input: true
- frame-subsampling-factor: 3
- right-tolerance: 5
- left-tolerance: 5
- exp/chain/tree
- exp/chain/0.trans_mdl
- ark:- ark:-

nnet3-chain-get-egs:

- srand: 0
- left-context: 29
- right-context: 29
- num-frames: 150,110,90
- frame-subsampling-factor: 3
- compress: true
- normalization-fst-scale: 1
- exp/chain/normalization.fst
