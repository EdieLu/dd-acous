#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

export CUDA_VISIBLE_DEVICES=3
# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6
# pytorch 1.1
source activate py13-cuda9

# ------------------------ hyper param --------------------------
max_seq_len=100 # 400 for char | 85 for word | 90 for bpe
minibatch_partition=50
batch_size=256

print_every=1
checkpoint_every=2
# print_every=150
# checkpoint_every=650
num_epochs=50
learning_rate=0.001

random_seed=25
eval_with_mask=True
savedir=acous-dd-models-v2/debug/

# ------------------------ file dir --------------------------
# -- asr --
# train_path_src=lib/add-acoustics/swbd/align/new_train_srctgt/asr/src.txt
# dev_path_src=lib/add-acoustics/swbd/align/new_valid_srctgt/asr/src.txt
# train_tsv_path=lib/add-acoustics/swbd/align/new_train_srctgt/asr/lab.tsv
# dev_tsv_path=lib/add-acoustics/swbd/align/new_valid_srctgt/asr/lab.tsv
# train_times_path=lib/add-acoustics/swbd/align/new_train_srctgt/asr/timestp.log
# dev_times_path=lib/add-acoustics/swbd/align/new_valid_srctgt/asr/timestp.log
# -- man --
train_path_src=lib/add-acoustics/swbd/align/new_train_srctgt/man/src.txt
dev_path_src=lib/add-acoustics/swbd/align/new_valid_srctgt/man/src.txt
train_tsv_path=lib/add-acoustics/swbd/align/new_train_srctgt/man/lab.tsv
dev_tsv_path=lib/add-acoustics/swbd/align/new_valid_srctgt/man/lab.tsv
train_times_path=lib/add-acoustics/swbd/align/new_train_srctgt/man-v2/timestp.log
dev_times_path=lib/add-acoustics/swbd/align/new_valid_srctgt/man-v2/timestp.log

train_acous_path=lib/add-acoustics/swbd/align/new_train_srctgt/asr/feat/flis
dev_acous_path=lib/add-acoustics/swbd/align/new_valid_srctgt/asr/feat/flis

# -- man-rmns --
# train_path_src=lib/add-acoustics/swbd/align/new_train_srctgt/man_rmns/src.txt
# dev_path_src=lib/add-acoustics/swbd/align/new_valid_srctgt/man_rmns/src.txt
# train_tsv_path=lib/add-acoustics/swbd/align/new_train_srctgt/man_rmns/lab.tsv
# dev_tsv_path=lib/add-acoustics/swbd/align/new_valid_srctgt/man_rmns/lab.tsv
# train_times_path=lib/add-acoustics/swbd/align/new_train_srctgt/man_rmns/timestp.log
# dev_times_path=lib/add-acoustics/swbd/align/new_valid_srctgt/man_rmns/timestp.log
# train_acous_path=lib/add-acoustics/swbd/align/new_train_srctgt/man_rmns/flis
# dev_acous_path=lib/add-acoustics/swbd/align/new_valid_srctgt/man_rmns/flis


tag_rev=True # default - True: E=1/O=0
keep_filler=True

use_type=word # word
path_vocab=lib/vocab/swbd-min1.en
load_embedding=lib/embeddings/glove.6B.200d.txt
embedding_size=200

# use_type=word # word-v2
# path_vocab=lib/vocab/clctotal+swbd.min-count4.en
# load_embedding=lib/embeddings/glove.6B.200d.txt
# embedding_size=200

# ------------------------ restart model --------------------------
loaddir=None

# ----------- [add timestamp]
# add_times=True
# add_acous=True

# restartdir=None
# batch_norm=False # default
# acous_norm=False
# spec_aug=False

# ----------- [not add timestamp]
# -- no las --
add_times=False
add_acous=False

restartdir=None
batch_norm=False
acous_norm=False
spec_aug=False

# -- word --
# add_times=False
# add_acous=True

# restartdir=acous-las-models-v3/las-word-v001/checkpoints/2020_04_14_03_40_19/
# batch_norm=False
# acous_norm=True
# spec_aug=True

# ------------------------ model config --------------------------
enc_mode=none # (las) pyramid | ts-pyramid | ts-collar (ts: used for timestamp mode only)
acous_hidden_size=256
acous_att_mode=bilinear # bilinear | bahdanau | hybrid
hidden_size_dec=200
hidden_size_shared=200
num_unilstm_dec=4

las_freeze=True
las_loss_weight=0.0
dd_loss_weight=1.0


export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/acous-dd-v2/train.py \
	--train_path_src $train_path_src \
	--train_acous_path $train_acous_path \
	--train_tsv_path $train_tsv_path \
	--train_times_path $train_times_path \
	--dev_path_src $dev_path_src \
	--dev_acous_path $dev_acous_path \
	--dev_tsv_path $dev_tsv_path \
	--dev_times_path $dev_times_path \
	--use_type $use_type \
	--path_vocab_src $path_vocab \
	--load_embedding $load_embedding \
	--add_times $add_times \
	--add_acous $add_acous \
	--keep_filler $keep_filler \
	--tag_rev $tag_rev \
	--save $savedir \
	--load $loaddir \
	--restart $restartdir \
	--random_seed $random_seed \
	--embedding_size $embedding_size \
	--acous_hidden_size $acous_hidden_size \
	--acous_att_mode $acous_att_mode \
	--hidden_size_dec $hidden_size_dec \
	--hidden_size_shared $hidden_size_shared \
	--num_unilstm_dec $num_unilstm_dec \
	--residual True \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--batch_first True \
	--eval_with_mask True \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--dropout 0.5 \
	--embedding_dropout 0.5 \
	--num_epochs $num_epochs \
	--use_gpu True \
	--max_grad_norm 1.0 \
	--learning_rate $learning_rate \
	--spec_aug $spec_aug \
	--acous_norm $acous_norm \
	--batch_norm $batch_norm \
	--enc_mode $enc_mode \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--minibatch_partition $minibatch_partition \
	--las_freeze $las_freeze \
	--las_loss_weight $las_loss_weight \
	--dd_loss_weight $dd_loss_weight \
