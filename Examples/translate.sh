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
source activate py13-cuda9

# ===========================================[vocab]===========================================
# use_type=char
# path_vocab=lib-bpe/vocab/char.en

use_type=word
path_vocab=lib/vocab/swbd-min1.en

# use_type=word
# path_vocab=lib/vocab/clctotal+swbd.min-count4.en

# ===========================================[path]============================================
# ------------------------------------------ word | char 
# -- man --
# fname=test_man_swbd_dd
# ftst=lib/add-acoustics/swbd/align/new_test_srctgt/man/src.txt
# acous_path=lib/add-acoustics/swbd/align/new_test_srctgt/asr/feat/flis
# tsv_path=lib/add-acoustics/swbd/align/new_test_srctgt/man/lab.tsv
# times_path=lib/add-acoustics/swbd/align/new_test_srctgt/man-v2/timestp.log
# # seqlen=360 #char
# seqlen=80 #word

# fname=test_man_eval3_dd
# ftst=lib/add-acoustics/eval3/test_srctgt/man/mansrc.txt
# acous_path=lib/add-acoustics/eval3/test_srctgt/feat/flis
# tsv_path=lib/add-acoustics/eval3/test_srctgt/man/manlab.tsv
# times_path=lib/add-acoustics/eval3/test_srctgt/man/man_timestp.log
# # seqlen=730 #char
# seqlen=150 #word

# -- asr --
# fname=test_swbd_dd
# ftst=lib/add-acoustics/swbd/align/new_test_srctgt/asr/src.txt
# acous_path=lib/add-acoustics/swbd/align/new_test_srctgt/asr/feat/flis
# tsv_path=lib/add-acoustics/swbd/align/new_test_srctgt/asr/lab.tsv
# times_path=lib/add-acoustics/swbd/align/new_test_srctgt/asr/timestp.log
# # seqlen=360 #char
# seqlen=80 #word

# fname=test_eval3_dd
# ftst=lib/add-acoustics/eval3/test_srctgt/src.txt
# acous_path=lib/add-acoustics/eval3/test_srctgt/feat/flis
# tsv_path=lib/add-acoustics/eval3/test_srctgt/lab.tsv
# times_path=lib/add-acoustics/eval3/test_srctgt/timestp.log
# # seqlen=730 #char
# seqlen=145 #word

# -------- EVAL3 V2 -----------
# --- with partial words ---
# fname=test_v2_man_eval3_dd
# ftst=lib/add-acoustics/eval3-segman/aln-res/man/src.txt
# acous_path=lib/add-acoustics/eval3-segman/aln-res/man/flis
# tsv_path=lib/add-acoustics/eval3-segman/aln-res/man/lab.tsv
# times_path=lib/add-acoustics/eval3-segman/aln-res/man/timestp.log
# seqlen=160 #word

# fname=test_v2_asr_eval3_dd #[same as v2nopw - can ignore]
# ftst=lib/add-acoustics/eval3-segman/aln-res/asr/src.txt.trim
# acous_path=lib/add-acoustics/eval3-segman/aln-res/asr/flis.trim
# tsv_path=lib/add-acoustics/eval3-segman/aln-res/asr/lab.tsv.trim
# times_path=lib/add-acoustics/eval3-segman/aln-res/asr/timestp.log.trim
# seqlen=160 #word

# --- without partial words ---
# fname=test_v2nopw_man_eval3_dd
# ftst=lib/add-acoustics/eval3-segman/aln-res-nopw/man/src.txt
# acous_path=lib/add-acoustics/eval3-segman/aln-res-nopw/man/flis
# tsv_path=lib/add-acoustics/eval3-segman/aln-res-nopw/man/lab.tsv
# times_path=lib/add-acoustics/eval3-segman/aln-res-nopw/man/timestp.log
# seqlen=160 #word

# fname=test_v2nopw_asr_eval3_dd
# ftst=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/src.txt.trim
# acous_path=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/flis.trim
# tsv_path=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/lab.tsv.trim
# times_path=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/timestp.log.trim
# seqlen=160 #word

# ------------------------ [dtal (matched)]
fname=test_dtal
ftst=lib/add-acoustics/eval3-segman/v2-aln/src.txt
acous_path=lib/add-acoustics/eval3-segman/v2-aln/flis
tsv_path=lib/add-acoustics/eval3-segman/v2-aln/lab.tsv
times_path=lib/add-acoustics/eval3-segman//v2-aln/timestp.log.aln
seqlen=160 #word

# fname=test_dtal_asr
# ftst=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/src.txt.trim
# acous_path=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/flis.trim
# tsv_path=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/lab.tsv.trim
# times_path=lib/add-acoustics/eval3-segman/aln-res-nopw/asr/timestp.log.trim
# seqlen=160 #word
# ===========================================[models]==========================================
# ------------------------------------------acous
# - word -
# add_acous=True
# add_times=False
# acous_norm=True
# keep_filler=True
# tag_rev=True # True: E=1/O=0
# model=acous-dd-models-v2/dd-v2-acous+las-v002/
# ckpt=2020_05_07_22_49_41
# batch_size=50

# ------------------------------------------noacous
# - word -
# add_acous=False
# add_times=False
# acous_norm=False
# keep_filler=True
# tag_rev=True # True: E=1/O=0
# model=acous-dd-models-v2/dd-v2-noacous-v002/
# ckpt=2020_05_07_21_56_03
# batch_size=200

# ------------------------------------------acous+ts
# - word -
add_acous=True
add_times=True
acous_norm=False
keep_filler=True
tag_rev=True # True: E=1/O=0
model=acous-dd-models-v2/dd-v2-acous+ts-v002/
ckpt=2020_05_08_01_49_54
batch_size=50


mode=2 # 2 | 6
use_gpu=True
# mode=6 # 2 | 6
# use_gpu=False

export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/acous-dd-v2/translate.py \
    --test_path_src $ftst \
    --path_vocab_src $path_vocab \
    --use_type $use_type \
    --test_acous_path $acous_path \
    --add_acous $add_acous \
    --acous_norm $acous_norm \
    --test_times_path $times_path \
    --add_times $add_times \
    --test_tsv_path $tsv_path \
    --tag_rev $tag_rev \
    --keep_filler $keep_filler \
    --load $model/checkpoints/$ckpt \
    --test_path_out $model/$fname/$ckpt/ \
    --max_seq_len $seqlen \
    --batch_size $batch_size \
    --use_gpu $use_gpu \
    --beam_width 1 \
    --eval_mode $mode
