import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.attention import AttentionLayer
from utils.config import PAD, EOS, BOS
from utils.dataset import load_pretrained_embedding, load_pretrained_embedding_bpe

import warnings
warnings.filterwarnings("ignore")

device = torch.device('cpu')

KEY_ATTN_SCORE = 'attention_score'
KEY_LENGTH = 'length'
KEY_SEQUENCE = 'sequence'
CLASSIFY_PROB = 'classify_prob'

class LAS(nn.Module):

	""" listen attend spell model + dd tag """

	def __init__(self,
		# params
		vocab_size,
		embedding_size=200,
		acous_hidden_size=256,
		acous_att_mode='bahdanau',
		hidden_size_dec=200,
		hidden_size_shared=200,
		num_unilstm_dec=4,
		#
		add_acous=True,
		acous_norm=False,
		spec_aug=False,
		batch_norm=False,
		enc_mode='pyramid',
		use_type='char',
		#
		add_times=False,
		#
		embedding_dropout=0,
		dropout=0.0,
		residual=True,
		batch_first=True,
		max_seq_len=32,
		load_embedding=None,
		word2id=None,
		id2word=None,
		hard_att=False,
		use_gpu=False
		):

		super(LAS, self).__init__()
		# config device
		if use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		# define model
		self.acous_dim = 40
		self.acous_hidden_size = acous_hidden_size
		self.acous_att_mode = acous_att_mode
		self.hidden_size_dec = hidden_size_dec
		self.hidden_size_shared = hidden_size_shared
		self.num_unilstm_dec = num_unilstm_dec

		# define var
		self.hard_att = hard_att
		self.residual = residual
		self.use_type = use_type
		self.max_seq_len = max_seq_len

		# tuning
		self.add_acous = add_acous
		self.acous_norm = acous_norm
		self.spec_aug = spec_aug
		self.batch_norm = batch_norm
		self.enc_mode = enc_mode

		# add time stamps
		self.add_times = add_times

		# use shared embedding + vocab
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.load_embedding = load_embedding
		self.word2id = word2id
		self.id2word = id2word

		# define operations
		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)

		# ------- load embeddings --------
		if self.use_type != 'bpe':
			if self.load_embedding:
				embedding_matrix = np.random.rand(
					self.vocab_size, self.embedding_size)
				embedding_matrix = load_pretrained_embedding(
					self.word2id, embedding_matrix, self.load_embedding)
				embedding_matrix = torch.FloatTensor(embedding_matrix)
				self.embedder = nn.Embedding.from_pretrained(embedding_matrix,
					freeze=False, sparse=False, padding_idx=PAD)
			else:
				self.embedder = nn.Embedding(self.vocab_size, self.embedding_size,
					sparse=False, padding_idx=PAD)
		elif self.use_type == 'bpe':
			# BPE
			embedding_matrix = np.random.rand(self.vocab_size, self.embedding_size)
			embedding_matrix = load_pretrained_embedding_bpe(embedding_matrix)
			embedding_matrix = torch.FloatTensor(embedding_matrix).to(device=device)
			self.embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False, padding_idx=PAD)

		# ------- las model --------
		if self.add_acous and not self.add_times:

			# ------ define acous enc -------
			if self.enc_mode == 'pyramid':
				self.acous_enc_l1 = torch.nn.LSTM(self.acous_dim,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)
				self.acous_enc_l2 = torch.nn.LSTM(self.acous_hidden_size * 4,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)
				self.acous_enc_l3 = torch.nn.LSTM(self.acous_hidden_size * 4,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)
				self.acous_enc_l4 = torch.nn.LSTM(self.acous_hidden_size * 4,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)
				if self.batch_norm:
					self.bn1 = nn.BatchNorm1d(self.acous_hidden_size * 2)
					self.bn2 = nn.BatchNorm1d(self.acous_hidden_size * 2)
					self.bn3 = nn.BatchNorm1d(self.acous_hidden_size * 2)
					self.bn4 = nn.BatchNorm1d(self.acous_hidden_size * 2)

			elif self.enc_mode == 'cnn':
				pass

			# ------ define acous att --------
			dropout_acous_att = dropout
			self.acous_hidden_size_att = 0 # ignored with bilinear

			self.acous_key_size = self.acous_hidden_size * 2 	# acous feats
			self.acous_value_size = self.acous_hidden_size * 2 	# acous feats
			self.acous_query_size = self.hidden_size_dec 		# use dec(words) as query
			self.acous_att = AttentionLayer(self.acous_query_size,
				self.acous_key_size, value_size=self.acous_value_size,
				mode=self.acous_att_mode, dropout=dropout_acous_att,
				query_transform=False, output_transform=False,
				hidden_size=self.acous_hidden_size_att, use_gpu=use_gpu,
				hard_att=False)

			# ------ define acous out --------
			self.acous_ffn = nn.Linear(
				self.acous_hidden_size * 2 + self.hidden_size_dec ,
				self.hidden_size_shared, bias=False)
			self.acous_out = nn.Linear(self.hidden_size_shared,
				self.vocab_size, bias=True)


			# ------ define acous dec -------
			# embedding_size_dec + self.hidden_size_shared [200+200] -> hidden_size_dec [200]
			if not self.residual:
				self.dec = torch.nn.LSTM(self.embedding_size + self.hidden_size_shared,
					self.hidden_size_dec, num_layers=self.num_unilstm_dec,
					batch_first=batch_first, bias=True, dropout=dropout,
					bidirectional=False)
			else:
				self.dec = nn.Module()
				self.dec.add_module('l0', torch.nn.LSTM(
					self.embedding_size + self.hidden_size_shared,
					self.hidden_size_dec, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=False))

				for i in range(1, self.num_unilstm_dec):
					self.dec.add_module('l'+str(i),
						torch.nn.LSTM(self.hidden_size_dec, self.hidden_size_dec,
						num_layers=1, batch_first=batch_first, bias=True,
						dropout=dropout, bidirectional=False))

		elif self.add_acous and self.add_times:

			# ------ define acous enc -------
			if self.enc_mode == 'ts-pyramid':
				self.acous_enc_l1 = torch.nn.LSTM(self.acous_dim,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)
				self.acous_enc_l2 = torch.nn.LSTM(self.acous_hidden_size * 4,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)
				self.acous_enc_l3 = torch.nn.LSTM(self.acous_hidden_size * 4,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)
				self.acous_enc_l4 = torch.nn.LSTM(self.acous_hidden_size * 4,
					self.acous_hidden_size, num_layers=1, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)

			else:
				# default
				acous_enc_blstm_depth = 1
				self.acous_enc = torch.nn.LSTM(self.acous_dim, self.acous_hidden_size,
					num_layers=acous_enc_blstm_depth, batch_first=batch_first,
					bias=True, dropout=dropout, bidirectional=True)

			# ------ define acous local att --------
			dropout_acous_att = dropout
			self.acous_hidden_size_att = 0 # ignored with bilinear

			self.acous_key_size = self.acous_hidden_size * 2 	# acous feats
			self.acous_value_size = self.acous_hidden_size * 2 	# acous feats
			self.acous_query_size = self.hidden_size_dec 		# use dec(words) as query
			self.acous_att = AttentionLayer(self.acous_query_size,
				self.acous_key_size, value_size=self.acous_value_size,
				mode=self.acous_att_mode, dropout=dropout_acous_att,
				query_transform=False, output_transform=False,
				hidden_size=self.acous_hidden_size_att, use_gpu=use_gpu,
				hard_att=False)

		# ------ define dd classifier -------
		self.dd_blstm_size = 300
		self.dd_blstm_depth = 2

		self.dd_blstm = torch.nn.LSTM(self.embedding_size, self.dd_blstm_size,
			num_layers=self.dd_blstm_depth, batch_first=batch_first, bias=True,
			dropout=dropout, bidirectional=True)
		if self.add_acous:
			dd_in_dim = self.dd_blstm_size * 2 + self.acous_hidden_size * 2
		else:
			dd_in_dim = self.dd_blstm_size * 2

		# might need to change this
		self.dd_classify = nn.Sequential(
				nn.Linear(dd_in_dim, 50, bias=True),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(50, 50),
				nn.LeakyReLU(0.2, inplace=True),
				nn.Linear(50, 1),
				nn.Sigmoid(),
		)
		# self.dd_classify = nn.Sequential(
		# 		nn.Linear(dd_in_dim, 50, bias=True),
		# 		nn.LeakyReLU(0.2, inplace=True),
		# 		nn.Linear(50, 1),
		# 		nn.Sigmoid(),
		# )

	def reset_max_seq_len(self, max_seq_len):

		self.max_seq_len = max_seq_len


	def set_idmap(self, word2id, id2word):

		self.word2id = word2id
		self.id2word = id2word


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None
			setattr(self, var_name, var_val)


	def pre_process_acous(self, acous_feats):

		"""
			acous_feats: b x max_time x max_channel
			spec-aug i.e. mask out certain time / channel
			time => t0 : t0 + t
			channel => f0 : f0 + f
		"""
		if not self.spec_aug:
			return acous_feats
		else:
			max_time = acous_feats.size(1)
			max_channel = 40

			CONST_MAXT_RATIO = 0.2
			CONST_T = int(min(40, CONST_MAXT_RATIO * max_time))
			CONST_F = int(7)
			REPEAT = 2

			for idx in range(REPEAT):

				t = random.randint(0, CONST_T)
				f = random.randint(0, CONST_F)
				t0 = random.randint(0, max_time-t-1)
				f0 = random.randint(0, max_channel-f-1)

				acous_feats[:,t0:t0+t,:] = 0
				acous_feats[:,:,f0:f0+f] = 0

			return acous_feats


	def forward(self, tgt, acous_feats=None, acous_times=None, hidden=None,
		is_training=False, teacher_forcing_ratio=0.0, beam_width=1, use_gpu=False):

		"""
			Args:
				src: list of acoustic features 	[b x acous_len x 40]
				tgt: list of word_ids 			[b x seq_len]
				hidden: initial hidden state
				is_training: whether in eval or train mode
				teacher_forcing_ratio: default at 1 - always teacher forcing
			Returns:
				decoder_outputs: list of step_output -
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
		"""

		# import pdb; pdb.set_trace()
		if use_gpu and torch.cuda.is_available():
			global device
			device = torch.device('cuda')
		else:
			device = torch.device('cpu')

		# if is_training:
		# 	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
		# else:
		# 	use_teacher_forcing = False
		use_teacher_forcing = True # always use reference text

		# --- 0. init var ----
		ret_dict = dict()
		ret_dict[KEY_ATTN_SCORE] = []

		decoder_outputs = []
		dec_hidden = None
		mask = None
		sequence_symbols = []
		batch_size = tgt.size(0)
		max_seq_len = tgt.size(1)
		lengths = np.array([max_seq_len] * batch_size)

		# ---- 1. convert id to embedding -----
		emb_tgt = self.embedding_dropout(self.embedder(tgt))

		self.check_var('add_times', False)
		if self.add_acous and not self.add_times:
			# ---- 2. run acous enc - pyramidal ----
			if is_training: acous_feats = self.pre_process_acous(acous_feats)
			acous_len = acous_feats.size(1)
			acous_hidden_init = None

			if self.enc_mode == 'pyramid':
				# layer1
				acous_outputs_l1, acous_hidden_l1 = self.acous_enc_l1(
					acous_feats, acous_hidden_init) # b x acous_len x 2dim
				acous_outputs_l1 = self.dropout(acous_outputs_l1)\
					.reshape(batch_size, acous_len, acous_outputs_l1.size(-1))
				if self.batch_norm:
					acous_outputs_l1 = self.bn1(acous_outputs_l1.permute(0, 2, 1))\
						.permute(0, 2, 1)
				acous_inputs_l2 = acous_outputs_l1.reshape(
					batch_size, int(acous_len/2), 2*acous_outputs_l1.size(-1))
					# b x acous_len/2 x 4dim

				# layer2
				acous_outputs_l2, acous_hidden_l2 = self.acous_enc_l2(
					acous_inputs_l2, acous_hidden_init) # b x acous_len/2 x 2dim
				acous_outputs_l2 = self.dropout(acous_outputs_l2)\
					.reshape(batch_size, int(acous_len/2), acous_outputs_l2.size(-1))
				if self.batch_norm:
						acous_outputs_l2 = self.bn2(acous_outputs_l2.permute(0, 2, 1))\
						.permute(0, 2, 1)
				acous_inputs_l3 = acous_outputs_l2\
					.reshape(batch_size, int(acous_len/4), 2*acous_outputs_l2.size(-1))
					# b x acous_len/4 x 4dim

				# layer3
				acous_outputs_l3, acous_hidden_l3 = self.acous_enc_l3(
					acous_inputs_l3, acous_hidden_init) # b x acous_len/4 x 2dim
				acous_outputs_l3 = self.dropout(acous_outputs_l3)\
					.reshape(batch_size, int(acous_len/4), acous_outputs_l3.size(-1))
				if self.batch_norm:
					acous_outputs_l3 = self.bn3(acous_outputs_l3.permute(0, 2, 1))\
						.permute(0, 2, 1)
				acous_inputs_l4 = acous_outputs_l3\
					.reshape(batch_size, int(acous_len/8), 2*acous_outputs_l3.size(-1)) # b x acous_len/8 x 4dim

				# layer4
				acous_outputs_l4, acous_hidden_l4 = self.acous_enc_l4(
					acous_inputs_l4, acous_hidden_init) # b x acous_len/8 x 2dim
				acous_outputs_l4 = self.dropout(acous_outputs_l4)\
					.reshape(batch_size, int(acous_len/8), acous_outputs_l4.size(-1))
				if self.batch_norm:
						acous_outputs_l4 = self.bn4(acous_outputs_l4.permute(0, 2, 1))\
							.permute(0, 2, 1)
				acous_outputs = acous_outputs_l4

			elif self.enc_mode == 'cnn':
				pass # todo

			# 3. ---- run dec + att + shared + output ----
			"""
				teacher_forcing_ratio = 1.0 -> always teacher forcing
				E.g.:
					acous 	        = [acous_len/8]
					tgt_chunk in    = w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
					predicted       = w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
			"""
			# import pdb; pdb.set_trace()

			att_keys = acous_outputs
			att_vals = acous_outputs

			# beam search decoding
			assert beam_width <= 1

			# no beam search decoding
			tgt_chunk = self.embedder(torch.Tensor([BOS]).repeat(batch_size,1)\
				.type(torch.LongTensor).to(device=device)) # BOS
			cell_value = torch.FloatTensor([0])\
				.repeat(batch_size, 1, self.hidden_size_shared).to(device=device)
			prev_c = torch.FloatTensor([0])\
				.repeat(batch_size, 1, max_seq_len).to(device=device)
			attn_outputs = []
			for idx in range(max_seq_len):
				predicted_logsoftmax, dec_hidden, step_attn, c_out, cell_value, attn_output = \
					self.forward_step(self.acous_att, self.acous_ffn, self.acous_out,
						att_keys, att_vals, tgt_chunk, cell_value,
						dec_hidden, mask, prev_c)
				predicted_logsoftmax = predicted_logsoftmax.squeeze(1) # [b, vocab_size]
				step_output = predicted_logsoftmax
				symbols, decoder_outputs, sequence_symbols, lengths = \
					self.decode(idx, step_output, decoder_outputs, sequence_symbols, lengths)
				prev_c = c_out
				if use_teacher_forcing:
					tgt_chunk = emb_tgt[:, idx].unsqueeze(1)
				else:
					tgt_chunk = self.embedder(symbols)
				ret_dict[KEY_ATTN_SCORE].append(step_attn)
				attn_outputs.append(attn_output)

			ret_dict[KEY_SEQUENCE] = sequence_symbols
			ret_dict[KEY_LENGTH] = lengths.tolist()
			attn_outputs = torch.cat(attn_outputs, dim=1)


		elif self.add_acous and self.add_times:

			# import pdb; pdb.set_trace()
			acous_len = acous_feats.size(1)
			acous_hidden_init = None

			# encode
			if self.enc_mode == 'ts-pyramid':

				"""
					try to mimic config in las model
					add collar on both sides of st/ed time
				"""

				if is_training: acous_feats = self.pre_process_acous(acous_feats)

				# layer1
				acous_outputs_l1, acous_hidden_l1 = \
					self.acous_enc_l1(acous_feats, acous_hidden_init) # b x acous_len x 2dim
				acous_outputs_l1 = self.dropout(acous_outputs_l1)\
					.reshape(batch_size, acous_len, acous_outputs_l1.size(-1))
				acous_inputs_l2 = acous_outputs_l1\
					.reshape(batch_size, int(acous_len/2), 2*acous_outputs_l1.size(-1))
					# b x acous_len/2 x 4dim

				# layer2
				acous_outputs_l2, acous_hidden_l2 = \
					self.acous_enc_l2(acous_inputs_l2, acous_hidden_init) # b x acous_len/2 x 2dim
				acous_outputs_l2 = self.dropout(acous_outputs_l2)\
					.reshape(batch_size, int(acous_len/2), acous_outputs_l2.size(-1))
				acous_inputs_l3 = acous_outputs_l2\
					.reshape(batch_size, int(acous_len/4), 2*acous_outputs_l2.size(-1))
					# b x acous_len/4 x 4dim

				# layer3
				acous_outputs_l3, acous_hidden_l3 = \
					self.acous_enc_l3(acous_inputs_l3, acous_hidden_init) # b x acous_len/4 x 2dim
				acous_outputs_l3 = self.dropout(acous_outputs_l3)\
					.reshape(batch_size, int(acous_len/4), acous_outputs_l3.size(-1))
				acous_inputs_l4 = acous_outputs_l3\
					.reshape(batch_size, int(acous_len/8), 2*acous_outputs_l3.size(-1))
					# b x acous_len/8 x 4dim

				# layer4
				acous_outputs_l4, acous_hidden_l4 = \
					self.acous_enc_l4(acous_inputs_l4, acous_hidden_init) # b x acous_len/8 x 2dim
				acous_outputs_l4 = self.dropout(acous_outputs_l4) \
					.reshape(batch_size, int(acous_len/8), acous_outputs_l4.size(-1))
				acous_outputs = acous_outputs_l4

				att_keys = acous_outputs # b x acous_len x 2dim
				att_vals = acous_outputs

				collect_att_out = []
				acous_times = torch.nn.utils.rnn.pad_sequence(
					[torch.LongTensor(elem) for elem in acous_times],
					batch_first=True) # b x l-1 x 2
				# adjust time - add collar, reduce by 8
				COLLAR = 10
				acous_times[:,:,0] = (acous_times[:,:,0] - COLLAR) / 8
				acous_times[:,:,1] = (acous_times[:,:,1] + COLLAR) / 8
				acous_times = torch.clamp(
					acous_times, min=0, max=acous_len/8).type(torch.long)
				acous_times = torch.cat((acous_times, torch.zeros((batch_size,1,2),
					dtype=torch.long)), dim=1) # include EOS - b x l x 2
				assert acous_times.size(1) == max_seq_len

				for lidx in range(max_seq_len):
					# b x acous_len
					mask = torch.zeros(acous_outputs.size()[:2], dtype=torch.bool)
					times = acous_times[:,lidx] # b x 2
					for bidx in range(batch_size):
						mask[bidx,:times[bidx,0]] = True # mask before start
						mask[bidx,times[bidx,1]:] = True # mask after end
					# import pdb; pdb.set_trace()
					att_qrys = emb_tgt[:,lidx] # b x h
					att_out, attn, c_out = self.acous_att(
						att_qrys, att_keys, att_vals, mask=mask.to(device))
					collect_att_out.append(att_out) # b x 2dim
					ret_dict[KEY_ATTN_SCORE].append(attn)

				# import pdb; pdb.set_trace()
				attn_outputs = torch.stack(collect_att_out, dim=1) # b x l x 2dim
				attn_outputs = self.dropout(attn_outputs)
				# -------------------------------------

			elif self.enc_mode == 'ts-collar':

				"""
					try to mimic config in las model
					add collar on both sides of st/ed time
				"""

				if is_training: acous_feats = self.pre_process_acous(acous_feats)
				# b x acous_len x 2dim
				acous_outputs, acous_hidden = self.acous_enc(acous_feats, acous_hidden_init)
				acous_outputs = self.dropout(acous_outputs)\
					.reshape(batch_size, acous_len, acous_outputs.size(-1))

				att_keys = acous_outputs # b x acous_len x 2dim
				att_vals = acous_outputs

				collect_att_out = []
				acous_times = torch.nn.utils.rnn.pad_sequence(
					[torch.LongTensor(elem) for elem in acous_times], batch_first=True)
					# b x l-1 x 2

				# adjust time - add collar
				COLLAR = 10
				acous_times[:,:,0] = (acous_times[:,:,0] - COLLAR)
				acous_times[:,:,1] = (acous_times[:,:,1] + COLLAR)
				acous_times = torch.clamp(acous_times, min=0, max=acous_len).type(torch.long)
				acous_times = torch.cat((acous_times, torch.zeros((batch_size,1,2),
					dtype=torch.long)), dim=1) # include EOS - b x l x 2
				assert acous_times.size(1) == max_seq_len

				for lidx in range(max_seq_len):
					mask = torch.zeros(acous_outputs.size()[:2], dtype=torch.bool) # b x acous_len
					times = acous_times[:,lidx] # b x 2
					for bidx in range(batch_size):
						mask[bidx,:times[bidx,0]] = True # mask before start
						mask[bidx,times[bidx,1]:] = True # mask after end
					# import pdb; pdb.set_trace()
					att_qrys = emb_tgt[:,lidx] # b x h
					att_out, attn, c_out = self.acous_att(
						att_qrys, att_keys, att_vals, mask=mask.to(device))
					collect_att_out.append(att_out) # b x 2dim
					ret_dict[KEY_ATTN_SCORE].append(attn)

				# import pdb; pdb.set_trace()
				attn_outputs = torch.stack(collect_att_out, dim=1) # b x l x 2dim
				attn_outputs = self.dropout(attn_outputs)
				# -------------------------------------

			else:
				acous_outputs, acous_hidden = self.acous_enc(
					acous_feats, acous_hidden_init) # b x acous_len x 2dim
				acous_outputs = self.dropout(acous_outputs)\
					.reshape(batch_size, acous_len, acous_outputs.size(-1))

				att_keys = acous_outputs # b x acous_len x 2dim
				att_vals = acous_outputs
				collect_att_out = []
				acous_times = torch.nn.utils.rnn.pad_sequence(
					[torch.LongTensor(elem) for elem in acous_times], batch_first=True)
					# b x l-1 x 2
				acous_times = torch.cat((acous_times, torch.zeros((batch_size,1,2),
					dtype=torch.long)), dim=1) # include EOS - b x l x 2
				assert acous_times.size(1) == max_seq_len

				for lidx in range(max_seq_len):
					mask = torch.zeros(acous_outputs.size()[:2], dtype=torch.bool)
					# b x acous_len
					times = acous_times[:,lidx] # b x 2
					for bidx in range(batch_size):
						mask[bidx,:times[bidx,0]] = True # mask before start
						mask[bidx,times[bidx,1]:] = True # mask after end
					# import pdb; pdb.set_trace()
					att_qrys = emb_tgt[:,lidx] # b x h
					att_out, attn, c_out = self.acous_att(
						att_qrys, att_keys, att_vals, mask=mask.to(device))
					collect_att_out.append(att_out) # b x 2dim
					ret_dict[KEY_ATTN_SCORE].append(attn)

				# import pdb; pdb.set_trace()
				attn_outputs = torch.stack(collect_att_out, dim=1) # b x l x 2dim
				attn_outputs = self.dropout(attn_outputs)
			# -------------------------------------

		# ---- 4. run dd -----
		# import pdb; pdb.set_trace()

		dd_blstm_hidden_init = None
		dd_blstm_outputs, dd_blstm_hidden = \
			self.dd_blstm(emb_tgt, dd_blstm_hidden_init)
		dd_blstm_outputs = self.dropout(dd_blstm_outputs)\
			.view(batch_size, max_seq_len, 2 * self.dd_blstm_size)

		if self.add_acous:
			attn = attn_outputs
			dd_cls_in = torch.cat( (dd_blstm_outputs, attn), dim = 2)
		else:
			dd_cls_in = dd_blstm_outputs

		dd_probs = self.dd_classify(dd_cls_in)
		ret_dict[CLASSIFY_PROB] = dd_probs


		return decoder_outputs, dec_hidden, ret_dict


	def decode(self, step, step_output, decoder_outputs, sequence_symbols, lengths):

			"""
				Greedy decoding
				Note:
					it should generate EOS, PAD as used in training tgt
				Args:
					step: step idx
					step_output: log predicted_softmax [batch_size, 1, vocab_size_dec]
				Returns:
					symbols: most probable symbol_id [batch_size, 1]
			"""
			decoder_outputs.append(step_output)
			symbols = decoder_outputs[-1].topk(1)[1]
			sequence_symbols.append(symbols)

			eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD))
			# equivalent to logical OR
			# eos_batches = symbols.data.eq(PAD)
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
			return symbols, decoder_outputs, sequence_symbols, lengths


	def forward_step(self, att_func, ffn_func, out_func,
		att_keys, att_vals, tgt_chunk, prev_cell_value,
		dec_hidden=None, mask_src=None, prev_c=None):

		"""
			manual unrolling - can only operate per time step

			Args:
				att_keys:   [batch_size, seq_len, acous_hidden_size * 2]
				att_vals:   [batch_size, seq_len, acous_hidden_size * 2]
				tgt_chunk:  tgt word embeddings
							non teacher forcing - [batch_size, 1, embedding_size_dec]
							(lose 1 dim when indexed)
				prev_cell_value:
							previous cell value before prediction
							[batch_size, 1, self.state_size]
				dec_hidden:
							initial hidden state for dec layer
				mask_src:
							mask of PAD for src sequences
				prev_c:
							used in hybrid attention mechanism

			Returns:
				predicted_softmax: log probilities [batch_size, vocab_size_dec]
				dec_hidden: a list of hidden states of each dec layer
				attn: attention weights
				cell_value: transformed attention output
					[batch_size, 1, self.hidden_size_shared]
		"""

		# record sizes
		batch_size = tgt_chunk.size(0)
		tgt_chunk_etd = torch.cat([tgt_chunk, prev_cell_value], -1)
		tgt_chunk_etd = tgt_chunk_etd\
			.view(-1, 1, self.embedding_size + self.hidden_size_shared)

		# run dec
		# default dec_hidden: [h_0, c_0];
		# with h_0 [num_layers * num_directions(==1), batch, hidden_size]
		if not self.residual:
			dec_outputs, dec_hidden = self.dec(tgt_chunk, dec_hidden)
			dec_outputs = self.dropout(dec_outputs)
		else:
			# store states layer by
			# layer num_layers * ([1, batch, hidden_size], [1, batch, hidden_size])
			dec_hidden_lis = []

			# layer0
			dec_func_first = getattr(self.dec, 'l0')
			if type(dec_hidden) == type(None):
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk_etd, None)
			else:
				index = torch.tensor([0]).to(device=device) # choose the 0th layer
				dec_hidden_in = tuple(
					[h.index_select(dim=0, index=index) for h in dec_hidden])
				dec_outputs, dec_hidden_out = \
					dec_func_first(tgt_chunk_etd, dec_hidden_in)
			dec_hidden_lis.append(dec_hidden_out)
			# no residual for 0th layer
			dec_outputs = self.dropout(dec_outputs)

			# layer1+
			for i in range(1, self.num_unilstm_dec):
				dec_inputs = dec_outputs
				dec_func = getattr(self.dec, 'l'+str(i))
				if type(dec_hidden) == type(None):
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, None)
				else:
					index = torch.tensor([i]).to(device=device)
					dec_hidden_in = tuple(
						[h.index_select(dim=0, index=index) for h in dec_hidden])
					dec_outputs, dec_hidden_out = \
						dec_func(dec_inputs, dec_hidden_in)
				dec_hidden_lis.append(dec_hidden_out)
				if i < self.num_unilstm_dec - 1:
					dec_outputs = dec_outputs + dec_inputs
				dec_outputs = self.dropout(dec_outputs)

			# convert to tuple
			h_0 = torch.cat([h[0] for h in dec_hidden_lis], 0)
			c_0 = torch.cat([h[1] for h in dec_hidden_lis], 0)
			dec_hidden = tuple([h_0, c_0])

		# run att
		att_func.set_mask(mask_src)
		att_outputs, attn, c_out = att_func(dec_outputs, att_keys, att_vals, prev_c=prev_c)
		att_outputs = self.dropout(att_outputs)

		# run ff + softmax
		ff_inputs = torch.cat((att_outputs, dec_outputs), dim=-1)
		ff_inputs_size = self.acous_hidden_size * 2 + self.hidden_size_dec
		cell_value = ffn_func(ff_inputs.view(-1, 1, ff_inputs_size)) # 600 -> 200
		outputs = out_func(cell_value.contiguous().view(-1, self.hidden_size_shared))
		predicted_logsoftmax = F.log_softmax(outputs, dim=1).view(batch_size, 1, -1)

		return predicted_logsoftmax, dec_hidden, attn, c_out, cell_value, att_outputs
