#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=1
echo $CUDA_VISIBLE_DEVICES 

# python 3.6
# pytorch 1.1
# source activate pt11-cuda9
source activate py13-cuda9

# ------------------------------------------------[files forgec]
fdir=acous-dd-models/res-collect

# fname=manclean-dd-noacous.flt.0.15
# fname=manclean-dd-wlas.flt.0.20
# fname=manclean-dd-wlas.flt.0.30
# fname=manclean-nodd.dsf
# fname=man-dd-noacous.flt.0.15.clean
# fname=man-dd-wlas.flt.0.20.clean
fname=man-dd-wlas.flt.0.30.clean
# fname=man-nodd.dsf
#fname=man-dd-wts.flt.0.12
# fname=dd-wts-pyramidlstm.flt.0.30

ftst=$fdir/$fname
seqlen=150

# ------------------------------------------------[models]
modeldir=acous-dd-models/gec
ckpt=16
ckptdir=checkpoints_epoch/$ckpt

export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/embedding-encdec-v6/translate.py \
    --test_path_src $ftst \
    --test_path_tgt $ftst \
    --path_vocab_src lib/vocab/clctotal+swbd.min-count4.en \
    --path_vocab_tgt lib/vocab/clctotal+swbd.min-count4.en \
    --load $modeldir/$ckptdir \
    --test_path_out $modeldir/new_results_acous_dd/eval3_$fname/$ckpt/ \
    --max_seq_len $seqlen \
    --batch_size 80 \
    --use_gpu True \
    --beam_width 1 \
    --eval_mode 2

